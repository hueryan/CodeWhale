//! Sanitized HTML rendering for assistant transcript messages.
//!
//! The web userscript this replaces handled site detection, DOM observers,
//! source toggles, script execution, and image export. The TUI path only keeps
//! the safe core pipeline:
//!
//! `LLM HTML response -> ammonia sanitizer -> scraper parser -> RichText IR -> ratatui lines`.

use ego_tree::NodeRef;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use scraper::Html;
use scraper::node::Node;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::palette;
use crate::tui::markdown_render::RenderedMarkdownLine;

const HTML_FENCE_PREFIXES: [&str; 3] = ["```html", "```HTML", "```Html"];

#[derive(Debug, Clone)]
struct RichSpan {
    text: String,
    style: Style,
}

#[derive(Debug, Clone, Default)]
struct RichLine {
    prefix: Vec<RichSpan>,
    segments: Vec<RichSpan>,
    continuation_indent: usize,
    is_pre: bool,
    force_blank: bool,
}

#[derive(Debug, Default)]
struct RenderContext {
    lines: Vec<RichLine>,
    current: RichLine,
    ordered_stack: Vec<usize>,
}

/// Render an assistant HTML response into ratatui text lines.
///
/// Returns `None` when the content does not look like an HTML response, so the
/// caller can keep using the Markdown renderer for ordinary assistant text.
#[must_use]
pub fn render_html_tagged(
    content: &str,
    width: u16,
    base_style: Style,
) -> Option<Vec<RenderedMarkdownLine>> {
    let source = extract_html_source(content)?;
    if !looks_like_html_response(source) {
        return None;
    }

    let sanitized = sanitize_html(source);
    let fragment = Html::parse_fragment(&sanitized);
    let mut ctx = RenderContext::default();

    for child in fragment.root_element().children() {
        walk_node(child, &mut ctx, base_style);
    }
    ctx.finish_current();

    let mut rendered = Vec::new();
    for line in ctx.lines {
        rendered.extend(wrap_rich_line(&line, width.max(1)));
    }

    if rendered.is_empty() {
        rendered.push(RenderedMarkdownLine {
            line: Line::from(""),
            is_code: false,
        });
    }

    Some(rendered)
}

fn extract_html_source(content: &str) -> Option<&str> {
    let trimmed = content.trim();
    for prefix in HTML_FENCE_PREFIXES {
        if let Some(rest) = trimmed.strip_prefix(prefix)
            && let Some(inner) = rest.strip_suffix("```")
        {
            return Some(inner.trim());
        }
    }
    Some(content)
}

fn looks_like_html_response(source: &str) -> bool {
    let trimmed = source.trim_start();
    if !trimmed.starts_with('<') {
        return false;
    }

    let lower = trimmed.to_ascii_lowercase();
    const RESPONSE_TAGS: [&str; 34] = [
        "!doctype",
        "html",
        "body",
        "main",
        "section",
        "article",
        "header",
        "footer",
        "nav",
        "aside",
        "div",
        "p",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "ul",
        "ol",
        "li",
        "table",
        "thead",
        "tbody",
        "tr",
        "th",
        "td",
        "blockquote",
        "pre",
        "code",
        "span",
        "a",
        "br",
        "hr",
    ];

    RESPONSE_TAGS
        .iter()
        .any(|tag| lower.starts_with(&format!("<{tag}")) || lower.contains(&format!("<{tag}")))
}

fn sanitize_html(source: &str) -> String {
    let mut cleaner = ammonia::Builder::default();
    cleaner.add_generic_attributes(["style"]);
    cleaner.clean(source).to_string()
}

fn walk_node(node: NodeRef<'_, Node>, ctx: &mut RenderContext, inherited_style: Style) {
    match node.value() {
        Node::Text(text) => ctx.push_text(text.text.as_ref(), inherited_style),
        Node::Element(element) => {
            let tag = element.name();
            match tag {
                "script" | "style" | "template" | "noscript" => {}
                "br" => ctx.finish_current_allow_blank(),
                "hr" => {
                    ctx.finish_current();
                    ctx.push_rule();
                }
                "h1" | "h2" | "h3" | "h4" | "h5" | "h6" => {
                    ctx.finish_current();
                    let style = merge_inline_style(
                        element.attr("style"),
                        inherited_style
                            .fg(palette::DEEPSEEK_SKY)
                            .add_modifier(Modifier::BOLD),
                    );
                    walk_children(node, ctx, style);
                    ctx.finish_current();
                }
                "p" | "div" | "section" | "article" | "header" | "footer" | "main" | "nav"
                | "aside" => {
                    ctx.finish_current();
                    let style = merge_inline_style(element.attr("style"), inherited_style);
                    walk_children(node, ctx, style);
                    ctx.finish_current();
                }
                "blockquote" => {
                    ctx.finish_current();
                    let previous = std::mem::take(&mut ctx.current.prefix);
                    ctx.current.prefix.push(RichSpan {
                        text: "│ ".to_string(),
                        style: Style::default().fg(palette::TEXT_DIM),
                    });
                    ctx.current.continuation_indent = 2;
                    walk_children(node, ctx, inherited_style.italic().fg(palette::TEXT_MUTED));
                    ctx.finish_current();
                    ctx.current.prefix = previous;
                }
                "ul" => {
                    ctx.finish_current();
                    walk_children(node, ctx, inherited_style);
                    ctx.finish_current();
                }
                "ol" => {
                    ctx.finish_current();
                    ctx.ordered_stack.push(1);
                    walk_children(node, ctx, inherited_style);
                    ctx.ordered_stack.pop();
                    ctx.finish_current();
                }
                "li" => {
                    ctx.finish_current();
                    let bullet = if let Some(next) = ctx.ordered_stack.last_mut() {
                        let bullet = format!("{next}. ");
                        *next += 1;
                        bullet
                    } else {
                        "• ".to_string()
                    };
                    ctx.current.prefix.push(RichSpan {
                        text: bullet.clone(),
                        style: Style::default().fg(palette::DEEPSEEK_SKY),
                    });
                    ctx.current.continuation_indent = bullet.width();
                    walk_children(node, ctx, inherited_style);
                    ctx.finish_current();
                }
                "pre" => {
                    ctx.finish_current();
                    push_pre_text(ctx, &text_content(node));
                }
                "table" => {
                    ctx.finish_current();
                    push_table(ctx, node, inherited_style);
                }
                "strong" | "b" => {
                    let style = merge_inline_style(
                        element.attr("style"),
                        inherited_style.add_modifier(Modifier::BOLD),
                    );
                    walk_children(node, ctx, style);
                }
                "em" | "i" => {
                    let style = merge_inline_style(
                        element.attr("style"),
                        inherited_style.add_modifier(Modifier::ITALIC),
                    );
                    walk_children(node, ctx, style);
                }
                "u" => {
                    let style = merge_inline_style(
                        element.attr("style"),
                        inherited_style.add_modifier(Modifier::UNDERLINED),
                    );
                    walk_children(node, ctx, style);
                }
                "code" | "kbd" | "samp" => {
                    let style = merge_inline_style(
                        element.attr("style"),
                        inherited_style
                            .fg(palette::DEEPSEEK_SKY)
                            .add_modifier(Modifier::ITALIC),
                    );
                    walk_children(node, ctx, style);
                }
                "a" => {
                    let style = merge_inline_style(
                        element.attr("style"),
                        inherited_style
                            .fg(palette::DEEPSEEK_BLUE)
                            .add_modifier(Modifier::UNDERLINED),
                    );
                    walk_children(node, ctx, style);
                    if let Some(href) = element.attr("href")
                        && !href.trim().is_empty()
                    {
                        ctx.push_text(
                            &format!(" ({href})"),
                            Style::default().fg(palette::TEXT_DIM),
                        );
                    }
                }
                "img" => {
                    let alt = element.attr("alt").unwrap_or("image");
                    ctx.push_text(
                        &format!("[image: {alt}]"),
                        Style::default().fg(palette::TEXT_DIM),
                    );
                }
                _ => {
                    let style = merge_inline_style(element.attr("style"), inherited_style);
                    walk_children(node, ctx, style);
                }
            }
        }
        _ => {}
    }
}

fn walk_children(node: NodeRef<'_, Node>, ctx: &mut RenderContext, style: Style) {
    for child in node.children() {
        walk_node(child, ctx, style);
    }
}

impl RenderContext {
    fn push_text(&mut self, text: &str, style: Style) {
        for word in text.split_whitespace() {
            if self.current_has_text() {
                self.current.segments.push(RichSpan {
                    text: " ".to_string(),
                    style,
                });
            }
            self.current.segments.push(RichSpan {
                text: word.to_string(),
                style,
            });
        }
    }

    fn current_has_text(&self) -> bool {
        !self.current.segments.is_empty()
    }

    fn finish_current(&mut self) {
        if self.current.force_blank
            || !self.current.prefix.is_empty()
            || !self.current.segments.is_empty()
        {
            self.lines.push(std::mem::take(&mut self.current));
        }
    }

    fn finish_current_allow_blank(&mut self) {
        if self.current.prefix.is_empty() && self.current.segments.is_empty() {
            self.current.force_blank = true;
        }
        self.finish_current();
    }

    fn push_rule(&mut self) {
        self.lines.push(RichLine {
            segments: vec![RichSpan {
                text: "─".repeat(40),
                style: Style::default().fg(palette::TEXT_DIM),
            }],
            ..RichLine::default()
        });
    }
}

fn push_pre_text(ctx: &mut RenderContext, text: &str) {
    let code_style = Style::default()
        .fg(palette::DEEPSEEK_SKY)
        .add_modifier(Modifier::ITALIC);
    for raw in text.trim_matches('\n').split('\n') {
        ctx.lines.push(RichLine {
            segments: vec![RichSpan {
                text: raw.to_string(),
                style: code_style,
            }],
            continuation_indent: 2,
            is_pre: true,
            ..RichLine::default()
        });
    }
}

fn push_table(ctx: &mut RenderContext, table: NodeRef<'_, Node>, base_style: Style) {
    let rows = table_rows(table);
    if rows.is_empty() {
        return;
    }
    let max_cols = rows.iter().map(Vec::len).max().unwrap_or(0);
    if max_cols == 0 {
        return;
    }
    let mut col_widths = vec![3usize; max_cols];
    for row in &rows {
        for (idx, cell) in row.iter().enumerate() {
            col_widths[idx] = col_widths[idx].max(cell.width().min(24));
        }
    }

    let sep_style = Style::default().fg(palette::TEXT_DIM);
    for row in rows {
        let mut line = RichLine::default();
        line.segments.push(RichSpan {
            text: "│ ".to_string(),
            style: sep_style,
        });
        for idx in 0..max_cols {
            let cell = row.get(idx).map(String::as_str).unwrap_or("");
            let width = col_widths[idx];
            let pad = width.saturating_sub(cell.width().min(width));
            line.segments.push(RichSpan {
                text: truncate_to_width(cell, width),
                style: base_style,
            });
            line.segments.push(RichSpan {
                text: " ".repeat(pad),
                style: base_style,
            });
            line.segments.push(RichSpan {
                text: if idx + 1 == max_cols { " │" } else { " │ " }.to_string(),
                style: sep_style,
            });
        }
        ctx.lines.push(line);
    }
}

fn table_rows(table: NodeRef<'_, Node>) -> Vec<Vec<String>> {
    let mut rows = Vec::new();
    collect_table_rows(table, &mut rows);
    rows
}

fn collect_table_rows(node: NodeRef<'_, Node>, rows: &mut Vec<Vec<String>>) {
    if element_name(node) == Some("tr") {
        let mut cells = Vec::new();
        collect_table_cells(node, &mut cells);
        if !cells.is_empty() {
            rows.push(cells);
        }
        return;
    }

    for child in node.children() {
        collect_table_rows(child, rows);
    }
}

fn collect_table_cells(node: NodeRef<'_, Node>, cells: &mut Vec<String>) {
    for child in node.children() {
        match element_name(child) {
            Some("th") | Some("td") => cells.push(
                text_content(child)
                    .split_whitespace()
                    .collect::<Vec<_>>()
                    .join(" "),
            ),
            _ => collect_table_cells(child, cells),
        }
    }
}

fn element_name(node: NodeRef<'_, Node>) -> Option<&str> {
    match node.value() {
        Node::Element(element) => Some(element.name()),
        _ => None,
    }
}

fn text_content(node: NodeRef<'_, Node>) -> String {
    let mut out = String::new();
    collect_text(node, &mut out);
    out
}

fn collect_text(node: NodeRef<'_, Node>, out: &mut String) {
    match node.value() {
        Node::Text(text) => out.push_str(text.text.as_ref()),
        Node::Element(element) if matches!(element.name(), "script" | "style" | "template") => {}
        _ => {
            for child in node.children() {
                collect_text(child, out);
            }
        }
    }
}

fn merge_inline_style(style_attr: Option<&str>, mut style: Style) -> Style {
    let Some(style_attr) = style_attr else {
        return style;
    };
    let lower = style_attr.to_ascii_lowercase();
    if lower.contains("font-weight:bold")
        || lower.contains("font-weight: bold")
        || lower.contains("font-weight:700")
    {
        style = style.add_modifier(Modifier::BOLD);
    }
    if lower.contains("font-style:italic") || lower.contains("font-style: italic") {
        style = style.add_modifier(Modifier::ITALIC);
    }
    if lower.contains("text-decoration:underline") || lower.contains("text-decoration: underline") {
        style = style.add_modifier(Modifier::UNDERLINED);
    }
    if lower.contains("color:") {
        if lower.contains("#60a5fa") || lower.contains("sky") || lower.contains("blue") {
            style = style.fg(palette::DEEPSEEK_BLUE);
        } else if lower.contains("#10b981") || lower.contains("green") {
            style = style.fg(Color::Green);
        } else if lower.contains("#f59e0b") || lower.contains("orange") || lower.contains("yellow")
        {
            style = style.fg(Color::Yellow);
        } else if lower.contains("#ef4444") || lower.contains("red") {
            style = style.fg(Color::Red);
        }
    }
    style
}

fn wrap_rich_line(line: &RichLine, width: u16) -> Vec<RenderedMarkdownLine> {
    if line.force_blank && line.segments.is_empty() && line.prefix.is_empty() {
        return vec![RenderedMarkdownLine {
            line: Line::from(""),
            is_code: false,
        }];
    }

    let width = usize::from(width.max(1));
    let mut out = Vec::new();
    let mut spans: Vec<Span<'static>> = line.prefix.iter().map(rich_span_to_span).collect();
    let mut current_width: usize = line.prefix.iter().map(|span| span.text.width()).sum();
    let continuation = " ".repeat(line.continuation_indent);

    for segment in &line.segments {
        let mut pending = String::new();
        let style = segment.style;
        for ch in segment.text.chars() {
            let ch_width = char_width(ch, current_width);
            if current_width + ch_width > width && (!spans.is_empty() || !pending.is_empty()) {
                if !pending.is_empty() {
                    spans.push(Span::styled(std::mem::take(&mut pending), style));
                }
                out.push(RenderedMarkdownLine {
                    line: Line::from(std::mem::take(&mut spans)),
                    is_code: line.is_pre,
                });
                if !continuation.is_empty() {
                    current_width = continuation.width();
                    spans.push(Span::raw(continuation.clone()));
                } else {
                    current_width = 0;
                }
                if !line.is_pre && ch.is_whitespace() {
                    continue;
                }
            }
            pending.push(ch);
            current_width += ch_width;
        }
        if !pending.is_empty() {
            spans.push(Span::styled(pending, style));
        }
    }

    if spans.is_empty() {
        spans.push(Span::raw(String::new()));
    }
    out.push(RenderedMarkdownLine {
        line: Line::from(spans),
        is_code: line.is_pre,
    });
    out
}

fn rich_span_to_span(span: &RichSpan) -> Span<'static> {
    Span::styled(span.text.clone(), span.style)
}

fn char_width(ch: char, col: usize) -> usize {
    match ch {
        '\t' => 8 - (col % 8),
        _ => ch.width().unwrap_or(1),
    }
}

fn truncate_to_width(text: &str, width: usize) -> String {
    let mut out = String::new();
    let mut used = 0usize;
    for ch in text.chars() {
        let w = ch.width().unwrap_or(1);
        if used + w > width {
            break;
        }
        out.push(ch);
        used += w;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn visible(lines: &[RenderedMarkdownLine]) -> Vec<String> {
        lines
            .iter()
            .map(|line| {
                line.line
                    .spans
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect::<String>()
            })
            .collect()
    }

    #[test]
    fn ignores_non_html_markdown() {
        assert!(render_html_tagged("# title\n\nplain markdown", 80, Style::default()).is_none());
    }

    #[test]
    fn renders_basic_html_fragment() {
        let lines = render_html_tagged(
            "<section><h2>Plan</h2><p>Hello <strong>world</strong>.</p><ul><li>one</li><li>two</li></ul></section>",
            80,
            Style::default(),
        )
        .expect("html should render");
        let text = visible(&lines).join("\n");
        assert!(text.contains("Plan"));
        assert!(text.contains("Hello world ."));
        assert!(text.contains("• one"));
        assert!(text.contains("• two"));
    }

    #[test]
    fn sanitizes_script_before_rendering() {
        let lines = render_html_tagged(
            "<div>safe<script>alert('x')</script><p>after</p></div>",
            80,
            Style::default(),
        )
        .expect("html should render");
        let text = visible(&lines).join("\n");
        assert!(text.contains("safe"));
        assert!(text.contains("after"));
        assert!(!text.contains("alert"));
    }

    #[test]
    fn renders_fenced_html_response() {
        let lines = render_html_tagged(
            "```html\n<div><p>boxed</p></div>\n```",
            80,
            Style::default(),
        )
        .expect("html fence should render");
        assert!(visible(&lines).join("\n").contains("boxed"));
    }
}

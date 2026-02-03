import type { MDXComponents } from 'mdx/types'

/**
 * MDX component overrides.
 *
 * SINGLE SOURCE OF TRUTH: All typography styling (font sizes, spacing)
 * is defined in globals.css via @theme tokens and .prose CSS rules.
 *
 * These components only add:
 * - scroll-m-20 for anchor link offset
 * - Semantic markup improvements
 * - Special handling for code blocks (rehype-pretty-code)
 */
export function useMDXComponents(components: MDXComponents): MDXComponents {
  return {
    ...components,

    // Headings - only add scroll margin for anchor links
    h1: ({ children }) => <h1 className="scroll-m-20">{children}</h1>,
    h2: ({ children }) => <h2 className="scroll-m-20">{children}</h2>,
    h3: ({ children }) => <h3 className="scroll-m-20">{children}</h3>,

    // Links
    a: ({ href, children }) => (
      <a href={href} className="font-medium">
        {children}
      </a>
    ),

    // Inline code only - code blocks are handled by rehype-pretty-code
    code: ({ children, ...props }) => {
      const isInlineCode = !('data-language' in props)
      if (!isInlineCode) return <code {...props}>{children}</code>
      return (
        <code className="relative rounded bg-muted px-[0.3rem] py-[0.2rem] font-mono text-sm font-semibold">
          {children}
        </code>
      )
    },

    // Code blocks - styled via globals.css, rehype-pretty-code handles syntax
    pre: ({ children, ...props }) => (
      <pre className="not-prose overflow-x-auto" {...props}>
        {children}
      </pre>
    ),

    // Tables
    table: ({ children }) => (
      <div className="my-6 w-full overflow-y-auto">
        <table className="w-full">{children}</table>
      </div>
    ),
    th: ({ children }) => (
      <th className="border px-4 py-2 text-left font-bold [&[align=center]]:text-center [&[align=right]]:text-right">
        {children}
      </th>
    ),
    td: ({ children }) => (
      <td className="border px-4 py-2 text-left [&[align=center]]:text-center [&[align=right]]:text-right">
        {children}
      </td>
    ),
  }
}

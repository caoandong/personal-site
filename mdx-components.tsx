import type { MDXComponents } from 'mdx/types'
import { cn } from '@/lib/utils'
import { textStyles } from '@/lib/typography'

export function useMDXComponents(components: MDXComponents): MDXComponents {
  return {
    ...components,

    // Override default HTML elements for consistent styling
    // Text styles derived from shared textStyles constant
    // Heading spacing handled by globals.css with adjacent sibling selectors
    // (* + h2) so first headings get no top margin automatically
    h1: ({ children }) => (
      <h1 className={cn('mb-8 scroll-m-20', textStyles.h1)}>
        {children}
      </h1>
    ),
    h2: ({ children }) => (
      <h2 className={cn('mb-6 scroll-m-20', textStyles.h2)}>
        {children}
      </h2>
    ),
    h3: ({ children }) => (
      <h3 className={cn('mb-4 scroll-m-20', textStyles.h3)}>
        {children}
      </h3>
    ),
    p: ({ children }) => (
      <p className="leading-relaxed my-6">{children}</p>
    ),
    ul: ({ children }) => (
      <ul className="my-8 ml-6 list-disc [&>li]:mt-3">{children}</ul>
    ),
    ol: ({ children }) => (
      <ol className="my-8 ml-6 list-decimal [&>li]:mt-3">{children}</ol>
    ),
    blockquote: ({ children }) => (
      <blockquote className="my-8 border-l-2 pl-6 italic">{children}</blockquote>
    ),
    a: ({ href, children }) => (
      <a
        href={href}
        className="font-medium"
      >
        {children}
      </a>
    ),
    code: ({ children }) => (
      <code className="relative rounded bg-muted px-[0.3rem] py-[0.2rem] font-mono text-sm font-semibold">
        {children}
      </code>
    ),
    pre: ({ children }) => (
      <pre className="not-prose mb-4 mt-6 overflow-x-auto rounded-lg border bg-muted p-4">
        {children}
      </pre>
    ),
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

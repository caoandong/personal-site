import { cva, type VariantProps } from 'class-variance-authority'

/**
 * Typography variants using CVA for consistent, type-safe text styling.
 *
 * Usage:
 *   import { typography } from '@/lib/typography'
 *   <h1 className={typography({ variant: 'h1' })}>Title</h1>
 *   <p className={typography({ variant: 'body', color: 'muted' })}>Text</p>
 */
export const typography = cva('', {
  variants: {
    variant: {
      // Headings
      h1: 'text-2xl md:text-5xl lg:text-6xl font-light tracking-tight leading-tight',
      h2: 'text-xl md:text-2xl font-light',
      h3: 'text-lg md:text-xl font-light',
      // Body text
      body: 'leading-relaxed',
      // Small text
      small: 'text-sm',
      caption: 'text-xs',
      // Navigation/UI
      nav: 'font-light',
    },
    color: {
      default: 'text-foreground',
      muted: 'text-grey-2',
      subtle: 'text-grey-1',
    },
    weight: {
      light: 'font-light',
      normal: 'font-normal',
      medium: 'font-medium',
      semibold: 'font-semibold',
      bold: 'font-bold',
    },
  },
  defaultVariants: {
    variant: 'body',
    color: 'default',
  },
})

export type TypographyVariants = VariantProps<typeof typography>

/**
 * Prose/article container styling for MDX content.
 * Uses Tailwind Typography plugin with custom overrides.
 */
export const prose = cva(
  'prose prose-xl prose-neutral dark:prose-invert max-w-none',
  {
    variants: {
      size: {
        default: 'prose-xl',
        lg: 'prose-lg',
        base: 'prose-base',
      },
    },
    defaultVariants: {
      size: 'default',
    },
  }
)

export type ProseVariants = VariantProps<typeof prose>

/**
 * Prose customizations for headings, paragraphs, etc.
 * Combine with the `prose` variant for full article styling.
 */
export const proseCustom = [
  // Headings
  'prose-headings:tracking-tight',
  'prose-h1:text-4xl prose-h1:font-bold prose-h1:mt-10',
  'prose-h2:text-3xl prose-h2:font-semibold prose-h2:mt-10 prose-h2:border-b prose-h2:pb-2',
  'prose-h3:text-2xl prose-h3:font-semibold prose-h3:mt-8',
  // Paragraphs
  'prose-p:leading-7 prose-p:mt-6',
  // Links
  'prose-a:text-foreground prose-a:no-underline hover:prose-a:text-grey-1',
  // Lists
  'prose-ul:my-6 prose-ol:my-6 prose-li:mt-2',
  // Blockquotes
  'prose-blockquote:mt-6 prose-blockquote:border-l-2 prose-blockquote:pl-6',
].join(' ')

/**
 * Container/layout constants for consistent page structure.
 */
export const layout = {
  container: 'mx-auto max-w-4xl px-6 pt-24 pb-16 md:pt-40 md:pb-24',
  header: 'flex items-center justify-between',
  headerEnd: 'flex items-center justify-end',
  headerSpacing: 'mb-16 md:mb-24',
} as const

export type LayoutKey = keyof typeof layout

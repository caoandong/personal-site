import { cva, type VariantProps } from 'class-variance-authority'

/**
 * Base heading styles - single source of truth for all heading typography.
 * Used by both the typography CVA and prose customizations.
 */
export const headingStyles = {
  h1: 'text-2xl leading-tight font-light tracking-tight md:text-5xl md:leading-tight lg:text-6xl',
  h2: 'text-xl font-light tracking-tight md:text-3xl',
  h3: 'text-lg font-light tracking-tight md:text-xl',
} as const

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
      // Headings (from shared constants)
      h1: headingStyles.h1,
      h2: headingStyles.h2,
      h3: headingStyles.h3,
      // Body text
      body: 'text-lg leading-relaxed',
      // Small text
      small: 'text-lg',
      caption: 'text-lg',
      // Navigation/UI
      nav: 'text-lg font-light',
    },
    color: {
      default: 'text-foreground',
      muted: 'text-grey-2',
      subtle: 'text-grey-1',
    },
    weight: {
      light: 'font-light',
      normal: 'font-normal',
      medium: 'font-normal',
      semibold: 'font-normal',
      bold: 'font-normal',
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
 * Converts a heading style string to prose-prefixed classes.
 * e.g., "text-2xl font-light md:text-5xl" → "prose-h1:text-2xl prose-h1:font-light md:prose-h1:text-5xl"
 */
function toProseHeading(tag: 'h1' | 'h2' | 'h3', styles: string): string {
  return styles
    .split(' ')
    .map((cls) => {
      // Handle responsive prefixes (md:, lg:, etc.)
      const match = cls.match(/^([a-z]+:)?(.+)$/)
      if (match) {
        const [, prefix = '', utility] = match
        return `${prefix}prose-${tag}:${utility}`
      }
      return `prose-${tag}:${cls}`
    })
    .join(' ')
}

/**
 * Prose customizations for headings, paragraphs, etc.
 * Combine with the `prose` variant for full article styling.
 * Heading styles are derived from headingStyles for consistency.
 */
export const proseCustom = [
  // Headings (derived from headingStyles)
  `${toProseHeading('h1', headingStyles.h1)} prose-h1:mt-10`,
  `${toProseHeading('h2', headingStyles.h2)} prose-h2:mt-10 prose-h2:border-b prose-h2:pb-2`,
  `${toProseHeading('h3', headingStyles.h3)} prose-h3:mt-8`,
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
  container: 'mx-auto max-w-4xl px-6 pt-24 pb-16 md:pt-24 md:pb-24',
  header: 'flex items-center justify-between',
  headerEnd: 'flex items-center justify-end',
  headerSpacing: 'mb-16 md:mb-24',
  /** Spacing below page title (h1) */
  titleSpacing: 'mb-8 md:mb-10',
  /** Spacing between content sections */
  sectionSpacing: 'space-y-8 md:space-y-10',
} as const

export type LayoutKey = keyof typeof layout

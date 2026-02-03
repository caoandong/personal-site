import { cva, type VariantProps } from 'class-variance-authority'

/**
 * Base text styles for JSX components (pages, MDX components).
 *
 * NOTE: Prose heading styles are defined in globals.css as CSS rules.
 * Keep both in sync when making changes!
 *
 * Mobile-first responsive sizing:
 * - h1: 2xl → 5xl (md) → 6xl (lg)
 * - h2: xl → 3xl (md)
 * - h3: xl → 2xl (md)
 */
export const textStyles = {
  // Headings
  h1: 'text-2xl leading-tight font-light tracking-tight md:text-5xl md:leading-tight lg:text-6xl',
  h2: 'text-xl font-light tracking-tight md:text-3xl',
  h3: 'text-xl font-light tracking-tight md:text-2xl',
  // Body text
  body: 'text-lg leading-relaxed',
  small: 'text-lg',
  caption: 'text-lg',
  nav: 'text-lg font-light',
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
      h1: textStyles.h1,
      h2: textStyles.h2,
      h3: textStyles.h3,
      // Body text (from shared constants)
      body: textStyles.body,
      small: textStyles.small,
      caption: textStyles.caption,
      nav: textStyles.nav,
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
 * Prose customizations for non-heading elements.
 *
 * SPACING VALUES (keep in sync with globals.css and mdx-components.tsx):
 * - Headings: h1 mt-12/mb-6, h2 mt-16/mb-4, h3 mt-12/mb-3
 * - Paragraphs: leading-relaxed (1.625), my-6
 * - Lists: my-8, list items mt-3
 * - Blockquotes: my-8
 *
 * Heading styles are defined in globals.css as the single source of truth.
 * Combine with the `prose` variant for full article styling.
 */
export const proseCustom = [
  // Paragraphs - line-height 1.625 (leading-relaxed) for optimal long-form readability
  'prose-p:leading-relaxed prose-p:my-6',
  // Links
  'prose-a:text-foreground prose-a:no-underline hover:prose-a:text-grey-1',
  // Lists - generous spacing for visual breathing room
  'prose-ul:my-8 prose-ol:my-8 prose-li:mt-3',
  // Blockquotes - clear visual separation
  'prose-blockquote:my-8 prose-blockquote:border-l-2 prose-blockquote:pl-6',
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

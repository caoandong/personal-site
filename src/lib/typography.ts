import { cva, type VariantProps } from 'class-variance-authority'

/**
 * Typography utilities that reference @theme tokens from globals.css
 *
 * SINGLE SOURCE OF TRUTH: All font sizes, line heights, and spacing
 * are defined in globals.css via @theme. This file provides:
 * - Tailwind utility classes for non-prose contexts
 * - CVA variants for type-safe styling
 *
 * To change typography, edit the @theme tokens in globals.css
 */

/**
 * Text style classes for non-prose contexts (pages, components).
 * These use CSS variables from @theme for consistency.
 */
export const textStyles = {
  // Headings - use CSS variables from @theme
  h1: 'text-[length:var(--font-size-h1)] leading-[var(--line-height-h1)] font-light tracking-tight md:text-[length:var(--font-size-h1-md)] lg:text-[length:var(--font-size-h1-lg)]',
  h2: 'text-[length:var(--font-size-h2)] leading-[var(--line-height-h2)] font-light tracking-tight md:text-[length:var(--font-size-h2-md)]',
  h3: 'text-[length:var(--font-size-h3)] leading-[var(--line-height-h3)] font-light tracking-tight md:text-[length:var(--font-size-h3-md)]',
  // Body text
  body: 'text-[length:var(--font-size-body)] leading-[var(--line-height-body)]',
  small: 'text-[length:var(--font-size-small)] leading-[var(--line-height-small)]',
  nav: 'text-[length:var(--font-size-nav)] leading-[var(--line-height-nav)] font-light',
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
      h1: textStyles.h1,
      h2: textStyles.h2,
      h3: textStyles.h3,
      body: textStyles.body,
      small: textStyles.small,
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
 * Uses Tailwind Typography plugin - all custom overrides are in globals.css
 */
export const prose = cva('prose prose-neutral dark:prose-invert max-w-none', {
  variants: {
    size: {
      default: 'prose-lg',
      lg: 'prose-xl',
      base: 'prose-base',
    },
  },
  defaultVariants: {
    size: 'default',
  },
})

export type ProseVariants = VariantProps<typeof prose>

/**
 * Container/layout constants for consistent page structure.
 */
export const layout = {
  container: 'mx-auto max-w-4xl px-6 pt-24 pb-16 md:pt-24 md:pb-24',
  header: 'flex items-center justify-between',
  headerEnd: 'flex items-center justify-end',
  headerSpacing: 'mb-16 md:mb-24',
  titleSpacing: 'mb-8 md:mb-10',
  sectionSpacing: 'space-y-8 md:space-y-10',
} as const

export type LayoutKey = keyof typeof layout

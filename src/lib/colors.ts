/**
 * SINGLE SOURCE OF TRUTH for all color constants.
 *
 * Color VALUES (hex codes) are defined in globals.css as CSS variables.
 * This file provides TypeScript constants for consistent Tailwind class usage.
 *
 * Naming follows Tailwind CSS conventions:
 * - Semantic names: foreground, background, muted, border, etc.
 * - Modifiers: text-{color}, bg-{color}, border-{color}
 */

// ============================================================================
// COLOR TOKENS
// ============================================================================

/**
 * Semantic color token names matching CSS variables in globals.css.
 * These are the canonical names for all colors in the design system.
 */
export const colorTokens = {
  // Core
  background: 'background',
  foreground: 'foreground',

  // Muted (subdued backgrounds and text)
  muted: 'muted',
  mutedForeground: 'muted-foreground',

  // Neutral scale (text hierarchy)
  neutral1: 'grey-1', // Subtle - used for hover states, secondary emphasis
  neutral2: 'grey-2', // Muted - used for tertiary text, placeholders

  // UI elements
  border: 'border',
  stroke: 'stroke',

  // Primary accent
  primary: 'primary',
  primaryForeground: 'primary-foreground',

  // Code highlighting
  codeHighlight: 'code-highlight',
  codeHighlightBorder: 'code-highlight-border',
} as const

export type ColorToken = (typeof colorTokens)[keyof typeof colorTokens]

// ============================================================================
// TEXT COLORS
// ============================================================================

/**
 * Text color classes for typography.
 * Use these for all text styling to ensure consistency.
 *
 * Hierarchy:
 * - default: Primary text, highest contrast
 * - muted: Secondary text, reduced emphasis
 * - subtle: Tertiary text, lowest emphasis (hover states, metadata)
 */
export const textColors = {
  default: 'text-foreground',
  muted: 'text-grey-2',
  subtle: 'text-grey-1',
  mutedForeground: 'text-muted-foreground',
  primary: 'text-primary',
  primaryForeground: 'text-primary-foreground',
  inherit: 'text-inherit',
} as const

export type TextColor = keyof typeof textColors
export type TextColorClass = (typeof textColors)[TextColor]

// ============================================================================
// BACKGROUND COLORS
// ============================================================================

/**
 * Background color classes for surfaces.
 */
export const bgColors = {
  default: 'bg-background',
  muted: 'bg-muted',
  primary: 'bg-primary',
  transparent: 'bg-transparent',
} as const

export type BgColor = keyof typeof bgColors
export type BgColorClass = (typeof bgColors)[BgColor]

// ============================================================================
// BORDER COLORS
// ============================================================================

/**
 * Border color classes for UI elements.
 * - default: Standard borders (inputs, cards)
 * - stroke: Divider lines, separators
 */
export const borderColors = {
  default: 'border-border',
  stroke: 'border-stroke',
  muted: 'border-muted',
  transparent: 'border-transparent',
} as const

export type BorderColor = keyof typeof borderColors
export type BorderColorClass = (typeof borderColors)[BorderColor]

// ============================================================================
// HOVER STATES
// ============================================================================

/**
 * Hover state color classes for interactive elements.
 */
export const hoverTextColors = {
  default: 'hover:text-foreground',
  subtle: 'hover:text-grey-1',
  muted: 'hover:text-grey-2',
} as const

export const hoverBgColors = {
  muted: 'hover:bg-muted',
  mutedSubtle: 'hover:bg-muted/50',
} as const

export type HoverTextColor = keyof typeof hoverTextColors
export type HoverBgColor = keyof typeof hoverBgColors

// ============================================================================
// GROUP HOVER STATES
// ============================================================================

/**
 * Group hover state classes for parent-triggered hover effects.
 */
export const groupHoverTextColors = {
  subtle: 'group-hover:text-grey-1',
  default: 'group-hover:text-foreground',
} as const

export type GroupHoverTextColor = keyof typeof groupHoverTextColors

// ============================================================================
// INTERACTIVE TEXT PATTERNS
// ============================================================================

/**
 * Common interactive text patterns combining base + hover states.
 * Use these for links, buttons, and other clickable text elements.
 */
export const interactiveText = {
  // Muted text that brightens on hover (nav links, metadata)
  mutedToSubtle: `${textColors.muted} ${hoverTextColors.subtle}`,
  // Subtle text that brightens to default on hover
  subtleToDefault: `${textColors.subtle} ${hoverTextColors.default}`,
  // Default text that dims on hover
  defaultToSubtle: `${textColors.default} ${hoverTextColors.subtle}`,
} as const

export type InteractiveText = keyof typeof interactiveText

// ============================================================================
// PROSE COLORS
// ============================================================================

/**
 * Prose-specific color classes for MDX/article content.
 * These map to Tailwind Typography plugin prose modifiers.
 */
export const proseColors = {
  body: 'prose-body:text-foreground',
  headings: 'prose-headings:text-foreground',
  links: 'prose-a:text-foreground',
  linksHover: 'hover:prose-a:text-grey-1',
  bold: 'prose-strong:text-foreground',
  code: 'prose-code:text-foreground',
  quotes: 'prose-blockquote:text-foreground',
  counters: 'prose-counters:text-grey-1',
  bullets: 'prose-bullets:text-grey-1',
  hr: 'prose-hr:border-stroke',
  quoteBorders: 'prose-blockquote:border-stroke',
} as const

export type ProseColor = keyof typeof proseColors

// ============================================================================
// TRANSITION UTILITIES
// ============================================================================

/**
 * Color transition classes for smooth state changes.
 */
export const colorTransitions = {
  default: 'transition-colors duration-200',
  fast: 'transition-colors duration-150',
  slow: 'transition-colors duration-300',
} as const

export type ColorTransition = keyof typeof colorTransitions

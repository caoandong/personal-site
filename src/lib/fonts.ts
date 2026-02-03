import localFont from 'next/font/local'
import { JetBrains_Mono } from 'next/font/google'

/**
 * Custom sans-serif font loaded from local woff2 files
 * Located in: src/fonts/custom-sans-serif/
 */
export const fontSans = localFont({
  src: [
    {
      path: '../fonts/custom-sans-serif/CustomFont-Light.woff2',
      weight: '300',
      style: 'normal',
    },
    {
      path: '../fonts/custom-sans-serif/CustomFont-LightItalic.woff2',
      weight: '300',
      style: 'italic',
    },
  ],
  variable: '--font-sans',
  display: 'swap',
})

/**
 * Monospace font for code blocks
 * Using Google Fonts (automatically self-hosted by Next.js)
 */
export const fontMono = JetBrains_Mono({
  subsets: ['latin'],
  variable: '--font-mono',
  display: 'swap',
})

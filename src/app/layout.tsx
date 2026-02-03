import type { Metadata } from 'next'
import { fontSans, fontMono } from '@/lib/fonts'
import { cn } from '@/lib/utils'
import { ThemeProvider } from '@/components/ThemeProvider'
import { NavigationProvider } from '@/components/NavigationProvider'
import './globals.css'

export const metadata: Metadata = {
  title: 'Personal Blog',
  description: 'A minimalistic personal blog with MDX, LaTeX, and interactive components',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={cn(
          fontSans.variable,
          fontMono.variable,
          'min-h-screen bg-background font-sans antialiased'
        )}
      >
        <ThemeProvider
          attribute="class"
          defaultTheme="system"
          enableSystem
          disableTransitionOnChange
        >
          <NavigationProvider>
            <div className="relative flex min-h-screen flex-col">
              <main className="flex-1">{children}</main>
            </div>
          </NavigationProvider>
        </ThemeProvider>
      </body>
    </html>
  )
}

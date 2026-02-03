import { ThemeToggle } from '@/components/ThemeToggle'
import { BackButton } from '@/components/BackButton'
import { cn } from '@/lib/utils'
import { layout } from '@/lib/typography'

export default function BlogLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <div className={layout.container}>
      <header className={cn(layout.header, layout.headerSpacing)}>
        <BackButton fallback="/" />
        <ThemeToggle />
      </header>
      {children}
    </div>
  )
}

import { ThemeToggle } from '@/components/ThemeToggle'
import { BackButton } from '@/components/BackButton'
import { Stagger } from '@/components/AnimateIn'
import { cn } from '@/lib/utils'
import { layout } from '@/lib/typography'

export default function BlogLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <Stagger className={cn(layout.container, layout.sectionSpacing)}>
      <header className={cn(layout.header)}>
        <BackButton fallback="/" />
        <ThemeToggle />
      </header>
      {children}
    </Stagger>
  )
}

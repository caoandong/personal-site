import { ThemeToggle } from '@/components/ThemeToggle'
import { BackButton } from '@/components/BackButton'
import { Stagger } from '@/components/AnimateIn'
import { layout } from '@/lib/typography'

export default function BlogLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <div className={layout.container}>
      <header className={layout.header}>
        <BackButton fallback="/" />
        <ThemeToggle />
      </header>
      <Stagger className={layout.sectionSpacing}>{children}</Stagger>
    </div>
  )
}

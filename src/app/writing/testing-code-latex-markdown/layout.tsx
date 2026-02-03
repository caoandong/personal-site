import { AnimateIn } from '@/components/AnimateIn'
import { prose } from '@/lib/typography'

export default function PostLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <AnimateIn as="article" stagger={1} className={prose()}>
      {children}
    </AnimateIn>
  )
}

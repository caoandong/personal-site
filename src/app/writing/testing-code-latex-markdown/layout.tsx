import { Stagger } from '@/components/AnimateIn'
import { prose } from '@/lib/typography'

export default function PostLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <Stagger as="article" className={prose()}>
      {children}
    </Stagger>
  )
}

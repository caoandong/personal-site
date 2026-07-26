import type { Metadata } from 'next'
import { Stagger } from '@/components/AnimateIn'
import { prose } from '@/lib/typography'

export const metadata: Metadata = {
  title: 'Automated worldbuilding with AI - Antonio Cao',
  description: 'Building automated systems that generate coherent worlds with AI.',
}

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

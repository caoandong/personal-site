import { prose } from '@/lib/typography'

export default function PostLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return <article className={prose()}>{children}</article>
}

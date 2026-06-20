import { NextResponse } from 'next/server'
import { posts } from '@/lib/posts'

export function middleware(request: Request & { nextUrl: URL }) {
  const match = new URL(request.url).pathname.match(/^\/writing\/([^/]+)/)
  if (match) {
    const slug = match[1]
    if (!posts.some((p) => p.slug === slug)) {
      return NextResponse.rewrite(new URL('/not-found', request.url))
    }
  }
  return NextResponse.next()
}

export const config = {
  matcher: '/writing/:slug*',
}

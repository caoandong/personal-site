export interface Post {
  slug: string
  title: string
  date: string
}

export const posts: Post[] = [
  {
    slug: 'testing-code-latex-markdown',
    title: 'Testing code and latex with markdown',
    date: '2025-02-02',
  },
]

export function getRecentPosts(limit?: number): Post[] {
  const sorted = [...posts].sort(
    (a, b) => new Date(b.date).getTime() - new Date(a.date).getTime()
  )
  return limit ? sorted.slice(0, limit) : sorted
}

export function groupPostsByYear(posts: Post[]): Record<string, Post[]> {
  return posts.reduce(
    (acc, post) => {
      const year = new Date(post.date).getFullYear().toString()
      if (!acc[year]) acc[year] = []
      acc[year].push(post)
      return acc
    },
    {} as Record<string, Post[]>
  )
}

export function formatDate(dateString: string): string {
  const date = new Date(dateString)
  const month = String(date.getMonth() + 1).padStart(2, '0')
  const day = String(date.getDate()).padStart(2, '0')
  return `${month}/${day}`
}

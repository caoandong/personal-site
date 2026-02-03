'use client'

import { useRouter } from 'next/navigation'
import { useCallback } from 'react'
import { cn } from '@/lib/utils'
import { typography } from '@/lib/typography'

interface BackButtonProps {
  fallback?: string
  className?: string
  children?: React.ReactNode
}

export function BackButton({ fallback = '/', className, children }: BackButtonProps) {
  const router = useRouter()

  const handleBack = useCallback(() => {
    // Check if there's history to go back to
    // window.history.state.idx tracks the position in Next.js history stack
    if (typeof window !== 'undefined' && window.history.state?.idx > 0) {
      router.back()
    } else {
      router.push(fallback)
    }
  }, [router, fallback])

  return (
    <button
      onClick={handleBack}
      className={cn(
        typography({ variant: 'nav', color: 'muted' }),
        'hover:text-foreground transition-colors duration-200 cursor-pointer',
        className
      )}
    >
      {children ?? <>&larr; Back</>}
    </button>
  )
}

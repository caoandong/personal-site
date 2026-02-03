'use client'

import { createContext, useContext, useEffect, useState, useCallback } from 'react'
import { usePathname } from 'next/navigation'
import { useRouter } from 'next/navigation'

interface NavigationContextType {
  canGoBack: boolean
  goBack: (fallback?: string) => void
}

const NavigationContext = createContext<NavigationContextType>({
  canGoBack: false,
  goBack: () => {},
})

const HISTORY_KEY = 'nav_history'

export function NavigationProvider({ children }: { children: React.ReactNode }) {
  const pathname = usePathname()
  const router = useRouter()
  const [canGoBack, setCanGoBack] = useState(false)

  useEffect(() => {
    if (typeof window === 'undefined') return

    // Try modern Navigation API first (most reliable)
    if ('navigation' in window) {
      const nav = window.navigation as { canGoBack: boolean }
      setCanGoBack(nav.canGoBack)
      return
    }

    // Fallback: sessionStorage-based tracking
    const historyJson = sessionStorage.getItem(HISTORY_KEY)
    const history: string[] = historyJson ? JSON.parse(historyJson) : []

    // Only add if it's a new navigation (not already the last item)
    if (history[history.length - 1] !== pathname) {
      history.push(pathname)
      sessionStorage.setItem(HISTORY_KEY, JSON.stringify(history))
    }

    // Can go back if we have more than one item in history
    setCanGoBack(history.length > 1)
  }, [pathname])

  const goBack = useCallback((fallback = '/') => {
    if (typeof window === 'undefined') {
      router.push(fallback)
      return
    }

    // Try modern Navigation API first
    if ('navigation' in window) {
      const nav = window.navigation as { canGoBack: boolean; back: () => void }
      if (nav.canGoBack) {
        router.back()
        return
      }
      router.push(fallback)
      return
    }

    // Fallback: sessionStorage-based tracking
    const historyJson = sessionStorage.getItem(HISTORY_KEY)
    const history: string[] = historyJson ? JSON.parse(historyJson) : []

    if (history.length > 1) {
      // Remove current page from history
      history.pop()
      sessionStorage.setItem(HISTORY_KEY, JSON.stringify(history))
      router.back()
    } else {
      // No history, go to fallback
      sessionStorage.setItem(HISTORY_KEY, JSON.stringify([fallback]))
      router.push(fallback)
    }
  }, [router])

  return (
    <NavigationContext.Provider value={{ canGoBack, goBack }}>
      {children}
    </NavigationContext.Provider>
  )
}

export function useNavigation() {
  return useContext(NavigationContext)
}

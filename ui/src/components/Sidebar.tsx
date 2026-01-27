'use client'

import Link from 'next/link'
import { usePathname, useRouter, useSearchParams } from 'next/navigation'
import { Suspense, useEffect, useState } from 'react'

const SidebarContent = () => {
  const pathname = usePathname()
  const router = useRouter()
  const searchParams = useSearchParams()
  const [mineral, setMineral] = useState(searchParams.get('mineral') || 'All Minerals')

  useEffect(() => {
    const currentMineral = searchParams.get('mineral') || 'All Minerals'
    setMineral(currentMineral)
  }, [searchParams])

  const handleMineralChange = (newMineral: string) => {
    setMineral(newMineral)
    const params = new URLSearchParams(searchParams.toString())
    if (newMineral === 'All Minerals') {
      params.delete('mineral')
    } else {
      params.set('mineral', newMineral)
    }
    const newUrl = `${pathname}?${params.toString()}`
    router.push(newUrl)
  }

  const links = [
    { href: '/', label: '🏠 Home' },
    { href: '/prediction', label: '🔮 Run Prediction' },
    { href: '/map', label: '🗺️ Map Visualization' },
    { href: '/stats', label: '📊 Statistics Dashboard' },
    { href: '/download', label: '📥 Download Results' },
  ]

  return (
    <div className="w-64 bg-gray-800 text-white h-screen fixed left-0 top-0 p-4">
      <h2 className="text-xl font-bold mb-8">AI Deposit Prediction</h2>

      <div className="mb-6">
        <label className="block text-sm font-medium mb-2">Mineral Focus:</label>
        <select
          value={mineral}
          onChange={(e) => handleMineralChange(e.target.value)}
          className="w-full p-2 bg-gray-700 border border-gray-600 rounded text-white"
        >
          <option value="All Minerals">All Minerals</option>
          <option value="Copper">Copper</option>
          <option value="Gold">Gold</option>
          <option value="Uranium">Uranium</option>
        </select>
      </div>

      <nav>
        <ul className="space-y-2">
          {links.map((link) => {
            const href = mineral === 'All Minerals' ? link.href : `${link.href}?mineral=${encodeURIComponent(mineral)}`
            return (
              <li key={link.href}>
                <Link
                  href={href}
                  className={`block p-2 rounded hover:bg-gray-700 ${pathname === link.href ? 'bg-gray-700' : ''
                    }`}
                >
                  {link.label}
                </Link>
              </li>
            )
          })}
        </ul>
      </nav>
    </div>
  )
}

const Sidebar = () => {
  return (
    <Suspense fallback={
      <div className="w-64 bg-gray-800 text-white h-screen fixed left-0 top-0 p-4">
        <h2 className="text-xl font-bold mb-8">AI Deposit Prediction</h2>
        <div className="animate-pulse">Loading...</div>
      </div>
    }>
      <SidebarContent />
    </Suspense>
  )
}

export default Sidebar
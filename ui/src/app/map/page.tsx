'use client'

import { useState, useEffect, Suspense } from 'react'
import { useSearchParams } from 'next/navigation'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Map as MapIcon, Maximize2, Layers } from 'lucide-react'

function MapContent() {
  const searchParams = useSearchParams()
  const [mapHtml, setMapHtml] = useState('')
  const [mineral, setMineral] = useState('All Minerals')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const mineralParam = searchParams.get('mineral') || 'All Minerals'
    setMineral(mineralParam)
  }, [searchParams])

  useEffect(() => {
    fetchMap()
  }, [mineral])

  const fetchMap = async () => {
    setLoading(true)
    try {
      const response = await fetch(`/api/map?mineral=${encodeURIComponent(mineral)}`)
      const data = await response.json()
      setMapHtml(data.map_html)
    } catch (error) {
      console.error(error)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex flex-col gap-6 p-6 md:p-8 lg:p-10 transition-all duration-300 min-h-screen">
      <header className="flex flex-col gap-2">
        <div className="flex items-center gap-2">
          <MapIcon className="w-8 h-8 text-emerald-600 dark:text-emerald-400" />
          <h1 className="text-3xl font-bold tracking-tight text-foreground">
            Map Visualization
          </h1>
        </div>
        <p className="text-muted-foreground text-lg">
          Interactive spatial overview of predicted deposits and geological features for <span className="text-foreground font-semibold underline decoration-emerald-500/50">{mineral}</span>.
        </p>
      </header>

      <Card className="flex-1 min-h-[600px] flex flex-col overflow-hidden border-emerald-500/10 shadow-xl shadow-emerald-500/5">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-4 bg-muted/30 border-b">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-emerald-500/10 rounded-lg">
              <Layers className="w-5 h-5 text-emerald-600" />
            </div>
            <div>
              <CardTitle className="text-xl">Interactive Deposit Map</CardTitle>
              <CardDescription>Predicted areas and known deposit locations</CardDescription>
            </div>
          </div>
          <button
            onClick={() => window.location.reload()}
            className="p-2 hover:bg-emerald-500/10 rounded-full transition-colors group"
            title="Refresh Map"
          >
            <Maximize2 className="w-5 h-5 text-muted-foreground group-hover:text-emerald-600" />
          </button>
        </CardHeader>
        <CardContent className="flex-1 p-0 relative bg-zinc-50 dark:bg-zinc-950">
          {loading ? (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 bg-background/80 backdrop-blur-sm z-50">
              <div className="w-12 h-12 border-4 border-emerald-500/20 border-t-emerald-500 rounded-full animate-spin" />
              <p className="text-sm font-medium text-muted-foreground animate-pulse">Loading spatial data...</p>
            </div>
          ) : mapHtml ? (
            <div
              className="w-full h-full min-h-[600px]"
              dangerouslySetInnerHTML={{ __html: mapHtml }}
            />
          ) : (
            <div className="flex flex-col items-center justify-center h-full gap-4 text-muted-foreground">
              <MapIcon className="w-16 h-16 opacity-10" />
              <p>No map data available for the current selection.</p>
            </div>
          )}
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="bg-muted/50 border-none">
          <CardContent className="pt-6">
            <p className="text-xs font-bold text-emerald-600 uppercase tracking-wider mb-1">Status</p>
            <p className="text-2xl font-bold">Live</p>
          </CardContent>
        </Card>
        <Card className="bg-muted/50 border-none">
          <CardContent className="pt-6">
            <p className="text-xs font-bold text-emerald-600 uppercase tracking-wider mb-1">Resolution</p>
            <p className="text-2xl font-bold">High Density</p>
          </CardContent>
        </Card>
        <Card className="bg-muted/50 border-none">
          <CardContent className="pt-6">
            <p className="text-xs font-bold text-emerald-600 uppercase tracking-wider mb-1">Region</p>
            <p className="text-2xl font-bold">Southern Africa</p>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

export default function Map() {
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <MapContent />
    </Suspense>
  )
}
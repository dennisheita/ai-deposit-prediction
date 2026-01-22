'use client'

import { useState, useEffect } from 'react'
import { useSearchParams } from 'next/navigation'
import Sidebar from '../../components/Sidebar'

export default function Map() {
  const searchParams = useSearchParams()
  const [mapHtml, setMapHtml] = useState('')
  const [mineral, setMineral] = useState('All Minerals')

  useEffect(() => {
    const mineralParam = searchParams.get('mineral') || 'All Minerals'
    setMineral(mineralParam)
  }, [searchParams])

  useEffect(() => {
    fetchMap()
  }, [mineral])

  const fetchMap = async () => {
    try {
      const response = await fetch(`/api/map?mineral=${encodeURIComponent(mineral)}`)
      const data = await response.json()
      setMapHtml(data.map_html)
    } catch (error) {
      console.error(error)
    }
  }

  return (
    <div className="flex">
      <Sidebar />
      <div className="ml-64 container mx-auto p-4">
        <h1 className="text-3xl font-bold mb-8">Map Visualization - {mineral}</h1>
        {mapHtml && (
          <div dangerouslySetInnerHTML={{ __html: mapHtml }} />
        )}
      </div>
    </div>
  )
}
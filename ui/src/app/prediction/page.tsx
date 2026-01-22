'use client'

import { useState, useEffect } from 'react'
import { useSearchParams } from 'next/navigation'
import Sidebar from '../../components/Sidebar'

export default function Prediction() {
  const searchParams = useSearchParams()
  const [file, setFile] = useState<File | null>(null)
  const [mineral, setMineral] = useState('All Minerals')
  const [threshold, setThreshold] = useState(0.5)
  const [mapHtml, setMapHtml] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    const mineralParam = searchParams.get('mineral') || 'All Minerals'
    setMineral(mineralParam)
  }, [searchParams])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!file) return

    setLoading(true)
    setError('')
    const formData = new FormData()
    formData.append('file', file)
    formData.append('mineral', mineral)
    formData.append('threshold', threshold.toString())

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData,
      })
      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Prediction failed')
      }
      const data = await response.json()
      setMapHtml(data.map_html)
    } catch (error: any) {
      setError(error.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex">
      <Sidebar />
      <div className="ml-64 container mx-auto p-4">
        <h1 className="text-3xl font-bold mb-8">Run Prediction - {mineral}</h1>
        <form onSubmit={handleSubmit} className="space-y-4 mb-8">
          <div>
            <label className="block mb-2">Upload Prediction Data:</label>
            <input
              type="file"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
              accept=".csv,.shp,.zip,.geojson"
              className="border p-2"
              required
            />
          </div>
          <div>
            <label className="block mb-2">Threshold: {threshold}</label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              className="w-full"
            />
          </div>
          <button
            type="submit"
            className="bg-purple-500 text-white px-4 py-2 rounded disabled:opacity-50"
            disabled={loading}
          >
            {loading ? 'Running...' : 'Run Prediction'}
          </button>
        </form>

        {error && <p className="text-red-500 mb-4">{error}</p>}

        {mapHtml && (
          <div>
            <h2 className="text-2xl font-bold mb-4">Prediction Results</h2>
            <div dangerouslySetInnerHTML={{ __html: mapHtml }} />
          </div>
        )}
      </div>
    </div>
  )
}
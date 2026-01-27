'use client'

import { useState, useEffect, Suspense } from 'react'
import { useSearchParams, useRouter } from 'next/navigation'
import FileUpload from '@/components/file-upload'
import { Label } from '@/components/ui/label'

function PredictionContent() {
  const searchParams = useSearchParams()
  const router = useRouter()
  const [file, setFile] = useState<File | null>(null)
  const [mineral, setMineral] = useState('All Minerals')
  const [threshold, setThreshold] = useState(0.5)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    const mineralParam = searchParams.get('mineral') || 'All Minerals'
    setMineral(mineralParam)
  }, [searchParams])

  const handleSubmit = async (e: React.FormEvent) => {
    if (e) e.preventDefault()
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

      // Redirect to map page instead of showing it here
      const params = new URLSearchParams()
      params.set('mineral', mineral)
      router.push(`/map?${params.toString()}`)

    } catch (error: any) {
      setError(error.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="container mx-auto p-4 flex flex-col items-center">
      <h1 className="text-3xl font-bold mb-8 w-full max-w-lg text-center">Run Prediction - {mineral}</h1>

      <FileUpload
        onFileSelect={setFile}
        onSubmit={handleSubmit}
        isSubmitting={loading}
        label="Upload Prediction Data"
        submitLabel="Run Prediction"
        acceptedTypesLabel="CSV, SHP, ZIP, GeoJSON, JSON"
        onCancel={() => {
          setFile(null)
          setError('')
        }}
      >
        <div className="space-y-2 text-left">
          <Label className="text-sm font-medium">Threshold: {threshold}</Label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={threshold}
            onChange={(e) => setThreshold(parseFloat(e.target.value))}
            className="w-full h-2 bg-muted rounded-lg appearance-none cursor-pointer accent-primary"
          />
        </div>
      </FileUpload>

      {error && <p className="text-red-500 mt-4 p-3 bg-red-50 rounded-lg text-sm font-medium w-full max-w-lg">{error}</p>}
    </div>
  )
}

export default function Prediction() {
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <PredictionContent />
    </Suspense>
  )
}
'use client'

import { useState } from 'react'
import Sidebar from '../../components/Sidebar'

export default function Upload() {
  const [file, setFile] = useState<File | null>(null)
  const [message, setMessage] = useState('')

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!file) return

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      })
      const data = await response.json()
      setMessage('File uploaded successfully')
    } catch (error) {
      setMessage('Upload failed')
    }
  }

  return (
    <div className="flex">
      <Sidebar />
      <div className="ml-64 container mx-auto p-4">
        <h1 className="text-3xl font-bold mb-8">Data Upload</h1>
        <form onSubmit={handleSubmit} className="space-y-4">
          <input
            type="file"
            onChange={(e) => setFile(e.target.files?.[0] || null)}
            accept=".shp,.zip,.csv,.geojson"
            className="border p-2"
            required
          />
          <button type="submit" className="bg-blue-500 text-white px-4 py-2 rounded">
            Upload
          </button>
        </form>
        {message && <p className="mt-4">{message}</p>}
      </div>
    </div>
  )
}
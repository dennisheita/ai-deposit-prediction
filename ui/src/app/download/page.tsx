'use client'

import { useState, useEffect } from 'react'
import { useSearchParams } from 'next/navigation'
import Sidebar from '../../components/Sidebar'

interface FileItem {
  0: number
  1: string
  2: string
  3: string
  4: string
  5: string
  6: string | null
}

export default function Download() {
  const searchParams = useSearchParams()
  const [files, setFiles] = useState<FileItem[]>([])
  const [mineral, setMineral] = useState('All Minerals')

  useEffect(() => {
    const mineralParam = searchParams.get('mineral') || 'All Minerals'
    setMineral(mineralParam)
  }, [searchParams])

  useEffect(() => {
    fetchFiles()
  }, [mineral])

  const fetchFiles = async () => {
    try {
      const response = await fetch(`/api/files?mineral=${encodeURIComponent(mineral)}`)
      const data = await response.json()
      setFiles(data.files)
    } catch (error) {
      console.error(error)
    }
  }

  const handleDownload = (fileId: number) => {
    window.open(`/api/download/${fileId}`, '_blank')
  }

  return (
    <div className="flex">
      <Sidebar />
      <div className="ml-64 container mx-auto p-4">
        <h1 className="text-3xl font-bold mb-8">Download Center - {mineral}</h1>
        <table className="w-full border-collapse border">
          <thead>
            <tr className="bg-gray-200">
              <th className="border p-2">ID</th>
              <th className="border p-2">Filename</th>
              <th className="border p-2">Type</th>
              <th className="border p-2">Upload Date</th>
              <th className="border p-2">Mineral</th>
              <th className="border p-2">Download</th>
            </tr>
          </thead>
          <tbody>
            {files.map((file) => (
              <tr key={file[0]}>
                <td className="border p-2">{file[0]}</td>
                <td className="border p-2">{file[1]}</td>
                <td className="border p-2">{file[3]}</td>
                <td className="border p-2">{file[4]}</td>
                <td className="border p-2">{file[6] || 'N/A'}</td>
                <td className="border p-2">
                  <button
                    onClick={() => handleDownload(file[0])}
                    className="bg-blue-500 text-white px-2 py-1 rounded"
                  >
                    Download
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
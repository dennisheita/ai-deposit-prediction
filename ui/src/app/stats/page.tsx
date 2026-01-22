'use client'

import { useState, useEffect } from 'react'
import { useSearchParams } from 'next/navigation'
import Sidebar from '../../components/Sidebar'

interface Model {
  0: number
  1: string
  2: string
  3: string
  4: string
}

interface Alert {
  0: number
  1: string
  2: string
  3: string
}

export default function Stats() {
  const searchParams = useSearchParams()
  const [models, setModels] = useState<Model[]>([])
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [report, setReport] = useState('')
  const [trendsImg, setTrendsImg] = useState('')
  const [fiImg, setFiImg] = useState('')
  const [mineral, setMineral] = useState('All Minerals')

  useEffect(() => {
    const mineralParam = searchParams.get('mineral') || 'All Minerals'
    setMineral(mineralParam)
  }, [searchParams])

  useEffect(() => {
    fetchStats()
  }, [mineral])

  const fetchStats = async () => {
    try {
      const response = await fetch(`/api/stats?mineral=${encodeURIComponent(mineral)}`)
      const data = await response.json()
      setModels(data.models)
      setAlerts(data.alerts)
      setReport(data.report)
      setTrendsImg(data.trends_img)
      setFiImg(data.fi_img)
    } catch (error) {
      console.error(error)
    }
  }

  return (
    <div className="flex">
      <Sidebar />
      <div className="ml-64 container mx-auto p-4">
        <h1 className="text-3xl font-bold mb-8">Statistics Dashboard - {mineral}</h1>

      {alerts.length > 0 && (
        <div className="mb-8">
          <h2 className="text-2xl font-bold mb-4">Active Alerts</h2>
          <ul className="list-disc pl-5">
            {alerts.map((alert) => (
              <li key={alert[0]} className="text-red-600">
                {alert[3]}: {alert[2]}
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="mb-8">
        <h2 className="text-2xl font-bold mb-4">Training Runs</h2>
        <p>Total: {models.length}</p>
        <table className="w-full border-collapse border">
          <thead>
            <tr className="bg-gray-200">
              <th className="border p-2">ID</th>
              <th className="border p-2">Version</th>
              <th className="border p-2">Performance Metrics</th>
              <th className="border p-2">Created Date</th>
              <th className="border p-2">Mineral</th>
            </tr>
          </thead>
          <tbody>
            {models.map((model) => (
              <tr key={model[0]}>
                <td className="border p-2">{model[0]}</td>
                <td className="border p-2">{model[1]}</td>
                <td className="border p-2">{model[2]}</td>
                <td className="border p-2">{model[3]}</td>
                <td className="border p-2">{model[4]}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="mb-8">
        <h2 className="text-2xl font-bold mb-4">Performance Report</h2>
        <pre className="whitespace-pre-wrap bg-gray-100 p-4 rounded">{report}</pre>
      </div>

      {trendsImg && (
        <div className="mb-8">
          <h2 className="text-2xl font-bold mb-4">Performance Trends</h2>
          <img src={`data:image/png;base64,${trendsImg}`} alt="Performance Trends" className="max-w-full" />
        </div>
      )}

      {fiImg && (
        <div className="mb-8">
          <h2 className="text-2xl font-bold mb-4">Feature Importance Evolution</h2>
          <img src={`data:image/png;base64,${fiImg}`} alt="Feature Importance Evolution" className="max-w-full" />
        </div>
      )}
      </div>
    </div>
  )
}
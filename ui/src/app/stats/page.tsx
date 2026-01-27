'use client'

import { useState, useEffect } from 'react'
import { useSearchParams, useRouter } from 'next/navigation'
import TrainingRunTable, { ModelData, TrainingStatus } from '@/components/TrainingRunTable'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { AlertCircle, TrendingUp, BarChart3, FileText, Zap } from 'lucide-react'

interface RawModel {
  0: number
  1: string
  2: string // performance_metrics (JSON string)
  3: string // created_date
  4: string // mineral
}

interface Alert {
  0: number
  1: string
  2: string
  3: string
}

export default function Stats() {
  const searchParams = useSearchParams()
  const router = useRouter()
  const [models, setModels] = useState<RawModel[]>([])
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [report, setReport] = useState('')
  const [trendsImg, setTrendsImg] = useState('')
  const [fiImg, setFiImg] = useState('')
  const [mineral, setMineral] = useState('All Minerals')
  const [allMinerals, setAllMinerals] = useState<string[]>([])

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
      setAllMinerals(data.all_minerals || [])
    } catch (error) {
      console.error(error)
    }
  }

  const modelData: ModelData[] = models.map((m) => {
    let metricsStr = 'N/A'
    try {
      if (m[2]) {
        const metrics = JSON.parse(m[2])
        if (metrics.auc) metricsStr = `AUC: ${metrics.auc.toFixed(3)}`
        else if (metrics.accuracy) metricsStr = `Acc: ${metrics.accuracy.toFixed(3)}`
        else metricsStr = Object.entries(metrics).map(([k, v]) => `${k}: ${v}`).join(', ')
      }
    } catch (e) {
      metricsStr = m[2] || 'N/A'
    }

    return {
      id: m[0],
      version: m[1],
      metrics: metricsStr,
      date: m[3],
      mineral: m[4] || 'Unknown',
      status: 'active' as TrainingStatus,
    }
  })

  const handleMineralChange = (newMineral: string) => {
    const params = new URLSearchParams(searchParams.toString())
    params.set('mineral', newMineral)
    router.push(`/stats?${params.toString()}`)
  }

  return (
    <div className="flex flex-col gap-6 p-6 md:p-8 lg:p-10 transition-all duration-300">
      <header className="flex flex-col gap-2">
        <div className="flex items-center gap-2">
          <BarChart3 className="w-8 h-8 text-blue-600 dark:text-blue-400" />
          <h1 className="text-3xl font-bold tracking-tight text-foreground">
            Statistics & Monitoring
          </h1>
        </div>
        <p className="text-muted-foreground text-lg">
          Comprehensive overview of model performance and training history for <span className="text-foreground font-semibold underline decoration-blue-500/50">{mineral}</span>.
        </p>
      </header>


      <div className="grid grid-cols-1 gap-6">
        <TrainingRunTable
          data={modelData}
          selectedMineral={mineral}
          onMineralChange={handleMineralChange}
          allMinerals={allMinerals}
        />
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <Card className="flex flex-col h-full overflow-hidden">
          <CardHeader className="flex flex-row items-center gap-3 space-y-0">
            <div className="p-2 bg-blue-500/10 rounded-lg">
              <FileText className="w-5 h-5 text-blue-600" />
            </div>
            <CardTitle>Performance Report</CardTitle>
          </CardHeader>
          <CardContent className="flex-1 overflow-auto bg-muted/30 p-0 border-t">
            <pre className="p-6 whitespace-pre-wrap font-mono text-sm leading-relaxed text-foreground/80">
              {report || "No report data available for the current selection."}
            </pre>
          </CardContent>
        </Card>

        {trendsImg && (
          <Card className="flex flex-col h-full overflow-hidden">
            <CardHeader className="flex flex-row items-center gap-3 space-y-0">
              <div className="p-2 bg-emerald-500/10 rounded-lg">
                <TrendingUp className="w-5 h-5 text-emerald-600" />
              </div>
              <CardTitle>Performance Trends</CardTitle>
            </CardHeader>
            <CardContent className="flex-1 flex items-center justify-center p-4 bg-white dark:bg-zinc-950 border-t">
              <img
                src={`data:image/png;base64,${trendsImg}`}
                alt="Performance Trends"
                className="w-full h-auto object-contain max-h-[400px]"
              />
            </CardContent>
          </Card>
        )}
      </div>

      {fiImg && (
        <Card className="overflow-hidden">
          <CardHeader className="flex flex-row items-center gap-3 space-y-0">
            <div className="p-2 bg-purple-500/10 rounded-lg">
              <BarChart3 className="w-5 h-5 text-purple-600" />
            </div>
            <div>
              <CardTitle>Feature Importance Evolution</CardTitle>
              <CardDescription>Impact of different indicators on model predictions over time</CardDescription>
            </div>
          </CardHeader>
          <CardContent className="p-6 bg-white dark:bg-zinc-950 border-t">
            <div className="flex justify-center">
              <img
                src={`data:image/png;base64,${fiImg}`}
                alt="Feature Importance Evolution"
                className="w-full max-w-5xl h-auto object-contain"
              />
            </div>
          </CardContent>
        </Card>
      )}

      <div className="h-4" aria-hidden="true" /> {/* Spacer */}
    </div>
  )
}
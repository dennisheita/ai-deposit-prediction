'use client'

import { useState, useEffect, Suspense } from 'react'
import { useSearchParams, useRouter } from 'next/navigation'
import TrainingRunTable, { ModelData, TrainingStatus } from '@/components/TrainingRunTable'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { TrendingUp, BarChart3, FileText, Zap, Activity, Wifi, WifiOff, Trophy } from 'lucide-react'
import { useTrainingWebSocket } from '@/hooks/useTrainingWebSocket'

interface RawModel {
  0: number
  1: string
  2: string // performance_metrics (JSON string)
  3: string // created_date
  4: string // mineral
}

function StatsContent() {
  const searchParams = useSearchParams()
  const router = useRouter()
  const [models, setModels] = useState<RawModel[]>([])
  const [report, setReport] = useState('')
  const [trendsImg, setTrendsImg] = useState('')
  const [fiImg, setFiImg] = useState('')
  const [mineral, setMineral] = useState('All Minerals')
  const [allMinerals, setAllMinerals] = useState<string[]>([])

  // WebSocket for live training status
  const { trainingState, isConnected } = useTrainingWebSocket()

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
      setReport(data.report)
      setTrendsImg(data.trends_img)
      setFiImg(data.fi_img)
      setAllMinerals(data.all_minerals || [])
    } catch (error) {
      console.error(error)
    }
  }

  // Calculate progress percentage
  const progressPercent = trainingState.total_runs > 0
    ? Math.round((trainingState.current_run / trainingState.total_runs) * 100)
    : 0

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
          {/* Live Training Indicator */}
          {trainingState.active && (
            <div className="flex items-center gap-2 ml-4 px-3 py-1 bg-green-100 dark:bg-green-900/30 rounded-full">
              <span className="relative flex h-3 w-3">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-3 w-3 bg-green-500"></span>
              </span>
              <span className="text-sm font-medium text-green-700 dark:text-green-400">Training Live</span>
            </div>
          )}
        </div>
        <div className="flex items-center gap-4">
          <p className="text-muted-foreground text-lg">
            Comprehensive overview of model performance and training history for <span className="text-foreground font-semibold underline decoration-blue-500/50">{mineral}</span>.
          </p>
          {/* WebSocket Connection Status */}
          <div className="flex items-center gap-2 text-sm">
            {isConnected ? (
              <>
                <Wifi className="w-4 h-4 text-green-500" />
                <span className="text-green-600">Live</span>
              </>
            ) : (
              <>
                <WifiOff className="w-4 h-4 text-red-500" />
                <span className="text-red-600">Offline</span>
              </>
            )}
          </div>
        </div>
      </header>

      {/* Live Training Status Card - Shows when training is active */}
      {trainingState.active && (
        <Card className="border-green-200 dark:border-green-800 bg-green-50/50 dark:bg-green-900/10">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-green-700 dark:text-green-400">
              <Activity className="w-5 h-5" />
              Live Training in Progress
              <span className="relative flex h-3 w-3">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-3 w-3 bg-green-500"></span>
              </span>
            </CardTitle>
            <CardDescription>
              Real-time updates from the training pipeline
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="space-y-1">
                <span className="text-sm text-muted-foreground">Current Mineral</span>
                <p className="text-lg font-semibold">{trainingState.mineral || '-'}</p>
              </div>
              <div className="space-y-1">
                <span className="text-sm text-muted-foreground">Progress</span>
                <p className="text-lg font-semibold">{trainingState.current_run} / {trainingState.total_runs}</p>
              </div>
              <div className="space-y-1">
                <span className="text-sm text-muted-foreground">Current Score</span>
                <p className="text-lg font-semibold text-blue-600">
                  {trainingState.current_score !== null && trainingState.current_score !== undefined
                    ? trainingState.current_score.toFixed(4)
                    : '-'}
                </p>
              </div>
              <div className="space-y-1">
                <span className="text-sm text-muted-foreground flex items-center gap-1">
                  <Trophy className="w-4 h-4 text-yellow-500" />
                  Best Score
                </span>
                <p className="text-lg font-semibold text-yellow-600">
                  {trainingState.best_score !== null && trainingState.best_score !== undefined
                    ? trainingState.best_score.toFixed(4)
                    : '-'}
                </p>
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Overall Progress</span>
                <span className="font-mono">{progressPercent}%</span>
              </div>
              <Progress value={progressPercent} className="h-2" />
            </div>

            {trainingState.status_message && (
              <div className="p-3 bg-white dark:bg-zinc-900 rounded-md text-sm border">
                <span className="font-medium">Status:</span> {trainingState.status_message}
              </div>
            )}
          </CardContent>
        </Card>
      )}

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

export default function Stats() {
  return (
    <Suspense fallback={
      <div className="flex flex-col gap-6 p-6 md:p-8 lg:p-10">
        <header className="flex flex-col gap-2">
          <div className="flex items-center gap-2">
            <BarChart3 className="w-8 h-8 text-blue-600 dark:text-blue-400" />
            <h1 className="text-3xl font-bold tracking-tight text-foreground">
              Statistics & Monitoring
            </h1>
          </div>
          <p className="text-muted-foreground text-lg">Loading...</p>
        </header>
      </div>
    }>
      <StatsContent />
    </Suspense>
  )
}
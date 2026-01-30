'use client'

import { useState, useEffect, Suspense } from 'react'
import { useSearchParams } from 'next/navigation'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Play, Square, Loader2, Zap, Activity, Wifi, WifiOff, Trophy, TrendingUp } from 'lucide-react'
import { useTrainingWebSocket } from '@/hooks/useTrainingWebSocket'

function BatchTrainingContent() {
    const searchParams = useSearchParams()
    const [mineral, setMineral] = useState(searchParams.get('mineral') || 'All Minerals')
    const [message, setMessage] = useState('')
    const [loading, setLoading] = useState(false)

    // Use WebSocket for real-time training status
    const { trainingState, isConnected, error } = useTrainingWebSocket()

    useEffect(() => {
        const mineralParam = searchParams.get('mineral') || 'All Minerals'
        setMineral(mineralParam)
    }, [searchParams])

    const startTraining = async () => {
        setLoading(true)
        setMessage('')
        try {
            const response = await fetch('/api/start_continuous_training', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: `mineral=${encodeURIComponent(mineral)}`,
            })
            const data = await response.json()
            setMessage(data.message)
        } catch (error) {
            setMessage('Failed to start training')
            console.error(error)
        } finally {
            setLoading(false)
        }
    }

    const stopTraining = async () => {
        setLoading(true)
        try {
            const response = await fetch('/api/stop_batch_training', {
                method: 'POST',
            })
            const data = await response.json()
            setMessage(data.message)
        } catch (error) {
            setMessage('Failed to stop training')
            console.error(error)
        } finally {
            setLoading(false)
        }
    }

    // Calculate progress percentage
    const progressPercent = trainingState.total_runs > 0
        ? Math.round((trainingState.current_run / trainingState.total_runs) * 100)
        : 0

    return (
        <div className="flex flex-col gap-6 p-6 md:p-8 lg:p-10 transition-all duration-300">
            <header className="flex flex-col gap-2">
                <div className="flex items-center gap-2">
                    <Zap className="w-8 h-8 text-yellow-500" />
                    <h1 className="text-3xl font-bold tracking-tight text-foreground">
                        Perpetual Training
                    </h1>
                </div>
                <p className="text-muted-foreground text-lg">
                    Run continuous model training for <span className="text-foreground font-semibold underline decoration-yellow-500/50">{mineral}</span>.
                </p>
            </header>

            {/* Connection Status */}
            <div className="flex items-center gap-2 text-sm">
                {isConnected ? (
                    <>
                        <Wifi className="w-4 h-4 text-green-500" />
                        <span className="text-green-600">Live updates connected</span>
                    </>
                ) : (
                    <>
                        <WifiOff className="w-4 h-4 text-red-500" />
                        <span className="text-red-600">Disconnected - reconnecting...</span>
                    </>
                )}
            </div>

            <div className="grid gap-6 md:grid-cols-2">
                <Card>
                    <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                            <Activity className="w-5 h-5" />
                            Training Status
                            {trainingState.active && (
                                <span className="relative flex h-3 w-3">
                                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                                    <span className="relative inline-flex rounded-full h-3 w-3 bg-green-500"></span>
                                </span>
                            )}
                        </CardTitle>
                        <CardDescription>
                            Real-time training status from WebSocket
                        </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        <div className="flex items-center justify-between">
                            <span className="text-sm font-medium">Status:</span>
                            <Badge variant={trainingState.active ? "default" : "secondary"}>
                                {trainingState.active ? '🔴 Running' : '⏸️ Idle'}
                            </Badge>
                        </div>

                        {trainingState.active && (
                            <div className="space-y-2">
                                <div className="flex justify-between text-sm">
                                    <span>Progress:</span>
                                    <span className="font-mono">{trainingState.current_run} / {trainingState.total_runs}</span>
                                </div>
                                <Progress value={progressPercent} className="h-2" />
                                <p className="text-xs text-muted-foreground text-right">{progressPercent}%</p>
                            </div>
                        )}

                        <div className="flex items-center justify-between">
                            <span className="text-sm font-medium">Current Mineral:</span>
                            <span className="text-sm font-semibold">{trainingState.mineral || '-'}</span>
                        </div>

                        {trainingState.current_score !== null && trainingState.current_score !== undefined && (
                            <div className="flex items-center justify-between">
                                <span className="text-sm font-medium">Current Score:</span>
                                <span className="text-sm font-mono text-blue-600">{trainingState.current_score.toFixed(4)}</span>
                            </div>
                        )}

                        {trainingState.best_score !== null && trainingState.best_score !== undefined && (
                            <div className="flex items-center justify-between bg-yellow-50 dark:bg-yellow-900/20 p-2 rounded">
                                <span className="text-sm font-medium flex items-center gap-1">
                                    <Trophy className="w-4 h-4 text-yellow-500" />
                                    Best Score:
                                </span>
                                <span className="text-sm font-mono font-bold text-yellow-600">{trainingState.best_score.toFixed(4)}</span>
                            </div>
                        )}

                        {trainingState.status_message && (
                            <div className="p-3 bg-muted rounded-md text-sm">
                                <span className="font-medium">Status:</span> {trainingState.status_message}
                            </div>
                        )}

                        {trainingState.last_update && (
                            <p className="text-xs text-muted-foreground">
                                Last update: {new Date(trainingState.last_update).toLocaleTimeString()}
                            </p>
                        )}
                    </CardContent>
                </Card>

                <Card>
                    <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                            <Play className="w-5 h-5" />
                            Control Panel
                        </CardTitle>
                        <CardDescription>
                            Start or stop the continuous training process
                        </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        <div className="flex flex-col gap-3">
                            <Button
                                onClick={startTraining}
                                disabled={trainingState.active || loading}
                                className="w-full"
                                size="lg"
                            >
                                {loading ? (
                                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                                ) : (
                                    <Play className="w-4 h-4 mr-2" />
                                )}
                                Start Perpetual Training
                            </Button>

                            <Button
                                onClick={stopTraining}
                                disabled={!trainingState.active || loading}
                                variant="destructive"
                                className="w-full"
                                size="lg"
                            >
                                {loading ? (
                                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                                ) : (
                                    <Square className="w-4 h-4 mr-2" />
                                )}
                                Stop Training
                            </Button>
                        </div>

                        {message && (
                            <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded-md text-sm text-blue-700 dark:text-blue-300">
                                {message}
                            </div>
                        )}

                        <div className="text-sm text-muted-foreground">
                            <p className="font-medium mb-1">What happens when you start:</p>
                            <ul className="list-disc list-inside space-y-1">
                                <li>Runs <code>python3 continuous_trainer.py --max-runs 5</code></li>
                                <li>Trains models for {mineral === 'All Minerals' ? 'Gold, Copper, Uranium' : mineral}</li>
                                <li>Uses Optuna hyperparameter optimization</li>
                                <li>Saves best models to the database</li>
                            </ul>
                        </div>
                    </CardContent>
                </Card>
            </div>

            {/* Recent Training History */}
            {trainingState.run_history && trainingState.run_history.length > 0 && (
                <Card>
                    <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                            <TrendingUp className="w-5 h-5" />
                            Recent Training History
                        </CardTitle>
                        <CardDescription>
                            Last {trainingState.run_history.length} training runs
                        </CardDescription>
                    </CardHeader>
                    <CardContent>
                        <div className="space-y-2">
                            {trainingState.run_history.map((run, idx) => (
                                <div
                                    key={idx}
                                    className={`flex items-center justify-between p-3 rounded-md text-sm ${run.success
                                            ? 'bg-green-50 dark:bg-green-900/20'
                                            : 'bg-red-50 dark:bg-red-900/20'
                                        }`}
                                >
                                    <div className="flex items-center gap-3">
                                        <span className="font-mono text-muted-foreground">#{run.run}</span>
                                        <span className="font-medium">{run.mineral}</span>
                                        {run.success ? (
                                            <Badge variant="outline" className="text-green-600 border-green-300">✓ Success</Badge>
                                        ) : (
                                            <Badge variant="outline" className="text-red-600 border-red-300">✗ Failed</Badge>
                                        )}
                                    </div>
                                    <div className="flex items-center gap-4">
                                        {run.score !== null && run.score !== undefined && (
                                            <span className="font-mono">Score: {run.score.toFixed(4)}</span>
                                        )}
                                        <span className="text-muted-foreground">{run.duration?.toFixed(1)}s</span>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </CardContent>
                </Card>
            )}

            <Card>
                <CardHeader>
                    <CardTitle>Training Details</CardTitle>
                    <CardDescription>
                        Information about the continuous training process
                    </CardDescription>
                </CardHeader>
                <CardContent>
                    <div className="space-y-4 text-sm">
                        <div>
                            <h4 className="font-semibold mb-2">Command Executed:</h4>
                            <code className="block p-3 bg-muted rounded-md font-mono text-xs">
                                python3 continuous_trainer.py --max-runs 5 --minerals {mineral === 'All Minerals' ? 'Gold Copper Uranium' : mineral}
                            </code>
                        </div>
                        <div>
                            <h4 className="font-semibold mb-2">Training Pipeline:</h4>
                            <ul className="list-disc list-inside space-y-1 text-muted-foreground">
                                <li>Advanced training with XGBoost, LightGBM, and Random Forest</li>
                                <li>Optuna hyperparameter optimization (50 trials per model)</li>
                                <li>Ensemble model creation via voting classifier</li>
                                <li>Stratified spatial negative sampling</li>
                                <li>Spatial cross-validation with GroupKFold</li>
                            </ul>
                        </div>
                    </div>
                </CardContent>
            </Card>
        </div>
    )
}

export default function BatchTraining() {
    return (
        <Suspense fallback={
            <div className="flex flex-col gap-6 p-6 md:p-8 lg:p-10">
                <header className="flex flex-col gap-2">
                    <div className="flex items-center gap-2">
                        <Zap className="w-8 h-8 text-yellow-500" />
                        <h1 className="text-3xl font-bold tracking-tight text-foreground">
                            Perpetual Training
                        </h1>
                    </div>
                    <p className="text-muted-foreground text-lg">Loading...</p>
                </header>
            </div>
        }>
            <BatchTrainingContent />
        </Suspense>
    )
}

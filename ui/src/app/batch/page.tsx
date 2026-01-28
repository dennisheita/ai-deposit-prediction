'use client'

import { useState, useEffect } from 'react'
import { useSearchParams } from 'next/navigation'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Play, Square, Loader2, Zap, Activity } from 'lucide-react'

export default function BatchTraining() {
    const searchParams = useSearchParams()
    const [mineral, setMineral] = useState(searchParams.get('mineral') || 'All Minerals')
    const [isTraining, setIsTraining] = useState(false)
    const [iterations, setIterations] = useState(0)
    const [message, setMessage] = useState('')
    const [loading, setLoading] = useState(false)

    useEffect(() => {
        const mineralParam = searchParams.get('mineral') || 'All Minerals'
        setMineral(mineralParam)
    }, [searchParams])

    // Poll for training status
    useEffect(() => {
        const interval = setInterval(async () => {
            try {
                const response = await fetch('/api/continuous_training_status')
                const data = await response.json()
                setIsTraining(data.active)
                setIterations(data.iterations)
            } catch (error) {
                console.error('Failed to fetch training status:', error)
            }
        }, 2000)

        return () => clearInterval(interval)
    }, [])

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
            setIsTraining(true)
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
            setIsTraining(false)
        } catch (error) {
            setMessage('Failed to stop training')
            console.error(error)
        } finally {
            setLoading(false)
        }
    }

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

            <div className="grid gap-6 md:grid-cols-2">
                <Card>
                    <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                            <Activity className="w-5 h-5" />
                            Training Status
                        </CardTitle>
                        <CardDescription>
                            Current state of the continuous training process
                        </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        <div className="flex items-center justify-between">
                            <span className="text-sm font-medium">Status:</span>
                            <Badge variant={isTraining ? "default" : "secondary"}>
                                {isTraining ? 'Running' : 'Idle'}
                            </Badge>
                        </div>
                        <div className="flex items-center justify-between">
                            <span className="text-sm font-medium">Target Mineral:</span>
                            <span className="text-sm">{mineral}</span>
                        </div>
                        <div className="flex items-center justify-between">
                            <span className="text-sm font-medium">Completed Iterations:</span>
                            <span className="text-sm font-mono">{iterations}</span>
                        </div>
                        {message && (
                            <div className="p-3 bg-muted rounded-md text-sm">
                                {message}
                            </div>
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
                                disabled={isTraining || loading}
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
                                disabled={!isTraining || loading}
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

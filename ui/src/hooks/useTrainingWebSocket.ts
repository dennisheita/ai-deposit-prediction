'use client'

import { useState, useEffect, useRef, useCallback } from 'react'

export interface TrainingState {
    active: boolean
    mineral: string
    current_run: number
    total_runs: number
    current_score: number | null
    best_score: number | null
    status_message: string
    last_update: string | null
    run_history: Array<{
        run: number
        mineral: string
        success: boolean
        score: number | null
        duration: number
        error?: string
    }>
}

const INITIAL_STATE: TrainingState = {
    active: false,
    mineral: '',
    current_run: 0,
    total_runs: 0,
    current_score: null,
    best_score: null,
    status_message: 'Idle',
    last_update: null,
    run_history: []
}

export function useTrainingWebSocket() {
    const [trainingState, setTrainingState] = useState<TrainingState>(INITIAL_STATE)
    const [isConnected, setIsConnected] = useState(false)
    const [error, setError] = useState<string | null>(null)
    const wsRef = useRef<WebSocket | null>(null)
    const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null)
    const reconnectAttemptsRef = useRef(0)
    const MAX_RECONNECT_ATTEMPTS = 5

    const connect = useCallback(() => {
        // Determine WebSocket URL based on current location
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
        const host = window.location.host
        // Use the API port (8000) for WebSocket, or same host if using proxy
        const wsUrl = process.env.NEXT_PUBLIC_WS_URL || `${protocol}//${host}/ws/training`

        console.log('Connecting to WebSocket:', wsUrl)

        try {
            const ws = new WebSocket(wsUrl)
            wsRef.current = ws

            ws.onopen = () => {
                console.log('WebSocket connected')
                setIsConnected(true)
                setError(null)
                reconnectAttemptsRef.current = 0
            }

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data)
                    console.log('WebSocket message received:', data)
                    setTrainingState(prev => ({
                        ...prev,
                        ...data
                    }))
                } catch (e) {
                    console.error('Failed to parse WebSocket message:', e)
                }
            }

            ws.onerror = (event) => {
                console.error('WebSocket error:', event)
                setError('WebSocket connection error')
                setIsConnected(false)
            }

            ws.onclose = () => {
                console.log('WebSocket disconnected')
                setIsConnected(false)
                wsRef.current = null

                // Attempt to reconnect
                if (reconnectAttemptsRef.current < MAX_RECONNECT_ATTEMPTS) {
                    reconnectAttemptsRef.current++
                    const delay = Math.min(1000 * Math.pow(2, reconnectAttemptsRef.current), 10000)
                    console.log(`Reconnecting in ${delay}ms (attempt ${reconnectAttemptsRef.current})`)

                    reconnectTimeoutRef.current = setTimeout(() => {
                        connect()
                    }, delay)
                } else {
                    setError('Max reconnection attempts reached. Please refresh the page.')
                }
            }
        } catch (e) {
            console.error('Failed to create WebSocket:', e)
            setError('Failed to create WebSocket connection')
        }
    }, [])

    const disconnect = useCallback(() => {
        if (reconnectTimeoutRef.current) {
            clearTimeout(reconnectTimeoutRef.current)
            reconnectTimeoutRef.current = null
        }

        if (wsRef.current) {
            wsRef.current.close()
            wsRef.current = null
        }
    }, [])

    const sendMessage = useCallback((message: string) => {
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            wsRef.current.send(message)
        } else {
            console.warn('WebSocket not connected, cannot send message')
        }
    }, [])

    // Connect on mount
    useEffect(() => {
        connect()

        // Cleanup on unmount
        return () => {
            disconnect()
        }
    }, [connect, disconnect])

    // Heartbeat to keep connection alive
    useEffect(() => {
        if (!isConnected) return

        const heartbeat = setInterval(() => {
            sendMessage('ping')
        }, 30000) // Send ping every 30 seconds

        return () => clearInterval(heartbeat)
    }, [isConnected, sendMessage])

    return {
        trainingState,
        isConnected,
        error,
        sendMessage,
        reconnect: connect
    }
}

export default useTrainingWebSocket

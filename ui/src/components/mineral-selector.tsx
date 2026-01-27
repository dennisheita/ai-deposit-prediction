"use client"

import * as React from "react"
import { usePathname, useRouter, useSearchParams } from "next/navigation"
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select"

export function MineralSelector() {
    const pathname = usePathname()
    const router = useRouter()
    const searchParams = useSearchParams()
    const [mineral, setMineral] = React.useState(searchParams.get('mineral') || 'All Minerals')

    React.useEffect(() => {
        const currentMineral = searchParams.get('mineral') || 'All Minerals'
        setMineral(currentMineral)
    }, [searchParams])

    const handleMineralChange = (newMineral: string) => {
        setMineral(newMineral)
        const params = new URLSearchParams(searchParams.toString())
        if (newMineral === 'All Minerals') {
            params.delete('mineral')
        } else {
            params.set('mineral', newMineral)
        }
        const newUrl = `${pathname}?${params.toString()}`
        router.push(newUrl)
    }

    return (
        <div className="flex flex-col gap-2">
            <label className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                Mineral Focus
            </label>
            <Select value={mineral} onValueChange={handleMineralChange}>
                <SelectTrigger className="w-full">
                    <SelectValue placeholder="Select Mineral" />
                </SelectTrigger>
                <SelectContent>
                    <SelectItem value="All Minerals">All Minerals</SelectItem>
                    <SelectItem value="Copper">Copper</SelectItem>
                    <SelectItem value="Gold">Gold</SelectItem>
                    <SelectItem value="Uranium">Uranium</SelectItem>
                </SelectContent>
            </Select>
        </div>
    )
}

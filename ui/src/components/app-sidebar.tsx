"use client"

import * as React from "react"
import { usePathname, useSearchParams } from "next/navigation"
import {
    IconAd2,
    IconBellRinging,
    IconCalendar,
    IconCalendarStats,
    IconListDetails,
    IconNews,
    IconNotebook,
    IconProgressCheck,
    IconSettingsCode,
} from "@tabler/icons-react"
import {
    LayoutDashboard,
    Package,
    Upload,
    Settings,
    Sparkles,
    Map,
    BarChart,
    Download,
    Zap,
} from "lucide-react"

import { Sidebar, SidebarContent } from "@/components/ui/sidebar"
import { NavCollapsible } from "./nav-collapsible"
import { NavFooter } from "./nav-footer"
import { NavHeader } from "./nav-header"
import { NavMain } from "./nav-main"
import { SidebarData } from "./types"
import { useTrainingWebSocket } from "@/hooks/useTrainingWebSocket"

// Live Training Indicator Component
function LiveTrainingIndicator() {
    const { trainingState, isConnected } = useTrainingWebSocket()

    if (!trainingState.active) return null

    return (
        <div className="px-3 py-2">
            <a
                href="/batch"
                className="flex items-center gap-2 px-3 py-2 rounded-md bg-green-100 dark:bg-green-900/30 hover:bg-green-200 dark:hover:bg-green-900/50 transition-colors"
            >
                <span className="relative flex h-3 w-3">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-3 w-3 bg-green-500"></span>
                </span>
                <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-green-700 dark:text-green-400 truncate">
                        Training {trainingState.mineral}
                    </p>
                    <p className="text-xs text-green-600 dark:text-green-500">
                        Run {trainingState.current_run}/{trainingState.total_runs}
                    </p>
                </div>
                <Zap className="w-4 h-4 text-green-500" />
            </a>
        </div>
    )
}

export function AppSidebar({ ...props }: React.ComponentProps<typeof Sidebar>) {
    const pathname = usePathname()
    const searchParams = useSearchParams()
    const mineral = searchParams.get('mineral') || 'All Minerals'

    const getHref = (base: string) => {
        if (mineral === 'All Minerals') return base
        return `${base}?mineral=${encodeURIComponent(mineral)}`
    }

    const navMain = [
        {
            id: "overview",
            title: "Home",
            url: getHref('/'),
            icon: LayoutDashboard,
            isActive: pathname === '/',
        },
        {
            id: "prediction",
            title: "Run Prediction",
            url: getHref('/prediction'),
            icon: Sparkles,
            isActive: pathname === '/prediction',
        },
        {
            id: "map",
            title: "Map Visualization",
            url: getHref('/map'),
            icon: Map,
            isActive: pathname === '/map',
        },
        {
            id: "stats",
            title: "Statistics",
            url: getHref('/stats'),
            icon: BarChart,
            isActive: pathname === '/stats',
        },
        {
            id: "batch",
            title: "Perpetual Training",
            url: getHref('/batch'),
            icon: Zap,
            isActive: pathname === '/batch',
        },
        {
            id: "download",
            title: "Download Results",
            url: getHref('/download'),
            icon: Download,
            isActive: pathname === '/download',
        },
    ]

    const data: SidebarData = {
        user: {
            name: "User",
            email: "user@example.com",
            avatar: "/avatar-placeholder.png",
        },
        navMain: navMain,
        navCollapsible: {
            favorites: [
                {
                    id: "design",
                    title: "Design",
                    href: "#",
                    color: "bg-green-400 dark:bg-green-300",
                },
                {
                    id: "development",
                    title: "Development",
                    href: "#",
                    color: "bg-blue-400 dark:bg-blue-300",
                },
            ],
            teams: [
                {
                    id: "engineering",
                    title: "Engineering",
                    icon: IconSettingsCode,
                },
                {
                    id: "marketing",
                    title: "Marketing",
                    icon: IconAd2,
                },
            ],
            topics: [
                {
                    id: "product-updates",
                    title: "Product Updates",
                    icon: Package,
                },
            ],
        },
    }

    return (
        <Sidebar collapsible="icon" {...props} variant="sidebar">
            <NavHeader data={data} />
            <SidebarContent>
                {/* Live Training Indicator - shows in sidebar when training is active */}
                <LiveTrainingIndicator />
                <NavMain items={data.navMain} />
                <NavCollapsible
                    favorites={data.navCollapsible.favorites}
                    teams={data.navCollapsible.teams}
                    topics={data.navCollapsible.topics}
                />
            </SidebarContent>
            <NavFooter user={data.user} />
        </Sidebar>
    )
}

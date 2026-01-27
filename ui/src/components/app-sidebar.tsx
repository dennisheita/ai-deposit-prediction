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
} from "lucide-react"

import { Sidebar, SidebarContent } from "@/components/ui/sidebar"
import { NavCollapsible } from "./nav-collapsible"
import { NavFooter } from "./nav-footer"
import { NavHeader } from "./nav-header"
import { NavMain } from "./nav-main"
import { SidebarData } from "./types"

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

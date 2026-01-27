"use client"

import Link from "next/link"
import {
    MoreHorizontal,
    Star,
} from "lucide-react"

import {
    SidebarGroup,
    SidebarGroupContent,
    SidebarGroupLabel,
    SidebarMenu,
    SidebarMenuButton,
    SidebarMenuItem,
} from "@/components/ui/sidebar"
import { SidebarData } from "./types"

export function NavCollapsible({
    favorites,
    teams,
    topics,
}: {
    favorites: SidebarData["navCollapsible"]["favorites"]
    teams: SidebarData["navCollapsible"]["teams"]
    topics: SidebarData["navCollapsible"]["topics"]
}) {
    return (
        <>
            <SidebarGroup className="group-data-[collapsible=icon]:hidden">
                <SidebarGroupLabel>Favorites</SidebarGroupLabel>
                <SidebarGroupContent>
                    <SidebarMenu>
                        {favorites.map((item) => (
                            <SidebarMenuItem key={item.id}>
                                <SidebarMenuButton asChild>
                                    <Link href={item.href} title={item.title}>
                                        <div
                                            className={`flex size-2 rounded-full ${item.color}`}
                                        />
                                        <span>{item.title}</span>
                                    </Link>
                                </SidebarMenuButton>
                            </SidebarMenuItem>
                        ))}
                    </SidebarMenu>
                </SidebarGroupContent>
            </SidebarGroup>

            <SidebarGroup className="group-data-[collapsible=icon]:hidden">
                <SidebarGroupLabel>Teams</SidebarGroupLabel>
                <SidebarMenu>
                    {teams.map((item) => (
                        <SidebarMenuItem key={item.id}>
                            <SidebarMenuButton asChild>
                                <a href="#">
                                    <item.icon />
                                    <span>{item.title}</span>
                                </a>
                            </SidebarMenuButton>
                        </SidebarMenuItem>
                    ))}
                </SidebarMenu>
            </SidebarGroup>

            <SidebarGroup className="group-data-[collapsible=icon]:hidden">
                <SidebarGroupLabel>Topics</SidebarGroupLabel>
                <SidebarMenu>
                    {topics.map((item) => (
                        <SidebarMenuItem key={item.id}>
                            <SidebarMenuButton asChild>
                                <a href="#">
                                    <item.icon />
                                    <span>{item.title}</span>
                                </a>
                            </SidebarMenuButton>
                        </SidebarMenuItem>
                    ))}
                </SidebarMenu>
            </SidebarGroup>
        </>
    )
}

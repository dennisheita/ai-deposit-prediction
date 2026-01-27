"use client"

import { SidebarMenu, SidebarMenuItem } from "@/components/ui/sidebar"
import { MineralSelector } from "./mineral-selector"
import { SidebarData } from "./types"

export function NavHeader({ data }: { data: SidebarData }) {
    return (
        <SidebarMenu>
            <SidebarMenuItem>
                <div className="flex flex-col gap-4 py-2">
                    <div className="flex items-center gap-2 px-2">
                        <div className="flex aspect-square size-8 items-center justify-center rounded-lg bg-sidebar-primary text-sidebar-primary-foreground">
                            <svg
                                xmlns="http://www.w3.org/2000/svg"
                                viewBox="0 0 24 24"
                                fill="none"
                                stroke="currentColor"
                                strokeWidth="2"
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                className="size-4"
                            >
                                <path d="M21 12V7H5a2 2 0 0 1 0-4h14v4" />
                                <path d="M3 5v14a2 2 0 0 0 2 2h16v-5" />
                                <path d="M18 12a2 2 0 0 0 0 4h4v-4Z" />
                            </svg>
                        </div>
                        <div className="grid flex-1 text-left text-sm leading-tight">
                            <span className="truncate font-semibold">AI Deposit Prediction</span>
                            <span className="truncate text-xs">System v1.0</span>
                        </div>
                    </div>
                    <div className="px-2">
                        <MineralSelector />
                    </div>
                </div>
            </SidebarMenuItem>
        </SidebarMenu>
    )
}

'use client';

import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
} from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from '@/components/ui/select';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card';
import { History, Filter } from 'lucide-react';

export type TrainingStatus = 'active' | 'pending' | 'discontinued' | 'on-hold';

export interface ModelData {
    id: number;
    version: string;
    metrics: string;
    date: string;
    mineral: string;
    status: TrainingStatus;
}

function getStatusBadge(status: TrainingStatus) {
    switch (status) {
        case 'active':
            return (
                <Badge
                    variant="outline"
                    className="border-0 bg-green-500/15 text-green-700 hover:bg-green-500/25 dark:bg-green-500/10 dark:text-green-400 dark:hover:bg-green-500/20"
                >
                    Active
                </Badge>
            );
        case 'pending':
            return (
                <Badge
                    variant="outline"
                    className="border-0 bg-amber-500/15 text-amber-700 hover:bg-amber-500/25 dark:bg-amber-500/10 dark:text-amber-300 dark:hover:bg-amber-500/20"
                >
                    Pending
                </Badge>
            );
        case 'discontinued':
            return (
                <Badge
                    variant="outline"
                    className="border-0 bg-rose-500/15 text-rose-700 hover:bg-rose-500/25 dark:bg-rose-500/10 dark:text-rose-400 dark:hover:bg-rose-500/20"
                >
                    Discontinued
                </Badge>
            );
        case 'on-hold':
            return (
                <Badge
                    variant="outline"
                    className="border-0 bg-blue-500/15 text-blue-700 hover:bg-blue-500/25 dark:bg-blue-500/10 dark:text-blue-400 dark:hover:bg-blue-500/20"
                >
                    On Hold
                </Badge>
            );
        default:
            return null;
    }
}

interface TrainingRunTableProps {
    data: ModelData[];
    selectedMineral: string;
    onMineralChange: (value: string) => void;
    allMinerals: string[];
}

export default function TrainingRunTable({ data, selectedMineral, onMineralChange, allMinerals }: TrainingRunTableProps) {
    // Combine minerals from current data (may be filtered) with all available minerals from API
    const displayMinerals = Array.from(new Set([...allMinerals, ...data.map(item => item.mineral)]))
        .filter(m => m && m !== 'Unknown' && m !== 'All Minerals')
        .sort();

    return (
        <Card className="overflow-hidden border-none shadow-none bg-transparent">
            <CardHeader className="flex flex-col gap-4 px-0 pb-6 md:flex-row md:items-center md:justify-between">
                <div className="flex items-center gap-3">
                    <div className="p-2 bg-primary/10 rounded-lg text-primary">
                        <History className="w-5 h-5" />
                    </div>
                    <div>
                        <CardTitle className="text-xl">Training History</CardTitle>
                        <CardDescription>
                            Comprehensive log of model versions and performance
                        </CardDescription>
                    </div>
                </div>
                <div className="flex items-center gap-3">
                    <div className="flex items-center gap-2 text-sm font-medium text-muted-foreground bg-muted/50 px-3 py-1.5 rounded-md border">
                        <Filter className="w-4 h-4" />
                        <span>Filter by Mineral</span>
                        <Select value={selectedMineral} onValueChange={onMineralChange}>
                            <SelectTrigger className="h-8 w-[160px] border-none bg-transparent focus:ring-0 font-bold p-0 text-foreground">
                                <SelectValue placeholder="All Minerals" />
                            </SelectTrigger>
                            <SelectContent align="end">
                                <SelectItem value="All Minerals">All Minerals</SelectItem>
                                {displayMinerals.map((m) => (
                                    <SelectItem key={m} value={m}>
                                        {m}
                                    </SelectItem>
                                ))}
                            </SelectContent>
                        </Select>
                    </div>
                </div>
            </CardHeader>

            <CardContent className="p-0 rounded-xl border bg-card overflow-hidden">
                <div className="overflow-x-auto overflow-y-auto max-h-[500px]">
                    <Table>
                        <TableHeader className="bg-muted/50 sticky top-0 z-10">
                            <TableRow className="hover:bg-transparent border-b">
                                <TableHead className="w-[100px] font-semibold py-4">ID</TableHead>
                                <TableHead className="font-semibold py-4">Version</TableHead>
                                <TableHead className="font-semibold py-4">Mineral</TableHead>
                                <TableHead className="font-semibold py-4">Status</TableHead>
                                <TableHead className="text-right font-semibold py-4">Performance</TableHead>
                                <TableHead className="text-right font-semibold py-4 pr-6">Created On</TableHead>
                            </TableRow>
                        </TableHeader>
                        <TableBody>
                            {data.length > 0 ? (
                                data.map((item) => (
                                    <TableRow key={item.id} className="group hover:bg-muted/30 transition-colors">
                                        <TableCell className="font-mono text-xs font-bold text-muted-foreground py-4">
                                            TR-{item.id.toString().padStart(4, '0')}
                                        </TableCell>
                                        <TableCell className="font-medium py-4">
                                            <div className="flex items-center gap-2">
                                                <span className="w-1.5 h-1.5 rounded-full bg-blue-500" />
                                                {item.version}
                                            </div>
                                        </TableCell>
                                        <TableCell className="text-muted-foreground py-4">
                                            {item.mineral}
                                        </TableCell>
                                        <TableCell className="py-4">
                                            {getStatusBadge(item.status)}
                                        </TableCell>
                                        <TableCell className="text-right py-4">
                                            <span className="inline-flex items-center px-2.5 py-1 rounded-md bg-blue-500/10 text-blue-700 dark:text-blue-400 font-mono text-xs font-bold">
                                                {item.metrics}
                                            </span>
                                        </TableCell>
                                        <TableCell className="text-right text-muted-foreground py-4 pr-6">
                                            {new Date(item.date).toLocaleDateString(undefined, {
                                                month: 'short',
                                                day: '2-digit',
                                                year: 'numeric'
                                            })}
                                        </TableCell>
                                    </TableRow>
                                ))
                            ) : (
                                <TableRow>
                                    <TableCell
                                        colSpan={6}
                                        className="h-32 text-center text-muted-foreground"
                                    >
                                        <div className="flex flex-col items-center gap-2">
                                            <History className="w-8 h-8 opacity-20" />
                                            <p>No training runs found for this selection.</p>
                                        </div>
                                    </TableCell>
                                </TableRow>
                            )}
                        </TableBody>
                    </Table>
                </div>
            </CardContent>
        </Card>
    );
}

"use client"

import { useState, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { FileSpreadsheet, Upload, X } from "lucide-react";

interface FileUploadProps {
    onFileSelect: (file: File | null) => void;
    onSubmit?: (e: React.FormEvent) => void;
    onCancel?: () => void;
    accept?: string;
    label?: string;
    submitLabel?: string;
    recommendedSize?: string;
    acceptedTypesLabel?: string;
    isSubmitting?: boolean;
}

export default function FileUpload({
    onFileSelect,
    onSubmit,
    onCancel,
    accept = ".xlsx,.xls,.csv,.geojson,.json",
    label = "File Upload",
    submitLabel = "Upload",
    recommendedSize = "10 MB",
    acceptedTypesLabel = "XLSX, XLS, CSV, GeoJSON, JSON",
    isSubmitting = false,
    children,
}: FileUploadProps & { children?: React.ReactNode }) {
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0] || null;
        setSelectedFile(file);
        onFileSelect(file);
    };

    const removeFile = () => {
        setSelectedFile(null);
        onFileSelect(null);
        if (fileInputRef.current) {
            fileInputRef.current.value = "";
        }
    };

    return (
        <div className="sm:mx-auto sm:max-w-lg flex flex-col p-6 w-full max-w-lg bg-card rounded-xl border shadow-sm">
            <form onSubmit={onSubmit}>
                <h3 className="text-lg font-semibold text-foreground">{label}</h3>
                <div className="mt-4 flex justify-center space-x-4 rounded-md border border-dashed border-input px-6 py-10 bg-muted/30">
                    <div className="sm:flex sm:items-center sm:gap-x-3 text-center sm:text-left">
                        <Upload
                            className="mx-auto h-8 w-8 text-muted-foreground sm:mx-0 sm:h-6 sm:w-6"
                            aria-hidden={true}
                        />
                        <div className="mt-4 flex text-sm leading-6 text-foreground sm:mt-0">
                            <p>Drag and drop or</p>
                            <Label
                                htmlFor="file-upload"
                                className="relative cursor-pointer rounded-sm pl-1 font-medium text-primary hover:underline hover:underline-offset-4"
                            >
                                <span>choose file</span>
                                <input
                                    id="file-upload"
                                    name="file-upload"
                                    type="file"
                                    className="sr-only"
                                    accept={accept}
                                    onChange={handleFileChange}
                                    ref={fileInputRef}
                                />
                            </Label>
                            <p className="pl-1">to upload</p>
                        </div>
                    </div>
                </div>
                <p className="mt-2 flex items-center justify-between text-xs leading-5 text-muted-foreground">
                    Recommended max. size: {recommendedSize}, Accepted file types: {acceptedTypesLabel}.
                </p>

                {selectedFile && (
                    <div className="relative mt-8 rounded-lg bg-muted p-3 border">
                        <div className="absolute right-1 top-1">
                            <Button
                                type="button"
                                variant="ghost"
                                size="sm"
                                className="rounded-sm p-2 text-muted-foreground hover:text-foreground"
                                aria-label="Remove"
                                onClick={removeFile}
                            >
                                <X className="size-4 shrink-0" aria-hidden={true} />
                            </Button>
                        </div>
                        <div className="flex items-center space-x-2.5">
                            <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-sm bg-background shadow-sm ring-1 ring-inset ring-input">
                                <FileSpreadsheet
                                    className="size-5 text-foreground"
                                    aria-hidden={true}
                                />
                            </span>
                            <div className="w-full">
                                <p className="text-xs font-medium text-foreground truncate max-w-[200px]">
                                    {selectedFile.name}
                                </p>
                                <p className="mt-0.5 flex justify-between text-xs text-muted-foreground">
                                    <span>{(selectedFile.size / (1024 * 1024)).toFixed(2)} MB</span>
                                    <span>Completed</span>
                                </p>
                            </div>
                        </div>
                    </div>
                )}

                {children && <div className="mt-6 space-y-4">{children}</div>}

                <div className="mt-8 flex items-center justify-end space-x-3">
                    <Button
                        type="button"
                        variant="outline"
                        onClick={onCancel}
                        className="whitespace-nowrap rounded-sm border border-input px-4 py-2 text-sm font-medium text-foreground shadow-sm hover:bg-accent hover:text-foreground"
                    >
                        Cancel
                    </Button>
                    <Button
                        type="submit"
                        variant="default"
                        disabled={!selectedFile || isSubmitting}
                        className="whitespace-nowrap rounded-sm bg-primary px-4 py-2 text-sm font-medium text-primary-foreground shadow-sm hover:bg-primary/90 disabled:opacity-50"
                    >
                        {isSubmitting ? "Processing..." : submitLabel}
                    </Button>
                </div>
            </form>
        </div>
    );
}


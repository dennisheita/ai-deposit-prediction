import { LucideIcon } from "lucide-react";
import { Icon } from "@tabler/icons-react";

export interface SidebarData {
    user: {
        name: string;
        email: string;
        avatar: string;
    };
    navMain: {
        id: string;
        title: string;
        url: string;
        icon: LucideIcon | Icon | React.ElementType;
        isActive?: boolean;
    }[];
    navCollapsible: {
        favorites: {
            id: string;
            title: string;
            href: string;
            color: string;
        }[];
        teams: {
            id: string;
            title: string;
            icon: LucideIcon | Icon | React.ElementType;
        }[];
        topics: {
            id: string;
            title: string;
            icon: LucideIcon | Icon | React.ElementType;
        }[];
    };
}

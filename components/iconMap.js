import {
  Brain,
  Bot,
  BarChart3,
  Cpu,
  GraduationCap,
  Map,
  Folder,
  Layers,
  Shield,
  Scale,
  Zap,
  FlaskConical,
  FileCheck,
  Package,
  BookOpen,
  Sparkles,
  FileText,
  Rocket,
  Library,
} from "lucide-react";

export const CATEGORY_ICONS = {
  brain: Brain,
  bot: Bot,
  chart: BarChart3,
  cpu: Cpu,
  graduation: GraduationCap,
  map: Map,
  folder: Folder,
  layers: Layers,
  shield: Shield,
  scale: Scale,
  zap: Zap,
  flask: FlaskConical,
  "file-check": FileCheck,
  package: Package,
  book: BookOpen,
  sparkles: Sparkles,
  file: FileText,
  rocket: Rocket,
  library: Library,
};

export function getIcon(name, fallback = Folder) {
  return CATEGORY_ICONS[name] || fallback;
}
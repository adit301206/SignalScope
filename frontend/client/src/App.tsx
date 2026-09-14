import { Toaster } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Route, Switch } from "wouter";
import ErrorBoundary from "./components/ErrorBoundary";
import { ThemeProvider } from "./contexts/ThemeContext";
import { SiteHeader, Footer } from "./components/SignalScopeUI";
import Home from "./pages/Home";
import Analyze from "./pages/Analyze";
import HowItWorks from "./pages/HowItWorks";
import About from "./pages/About";
import { HistoryPage, InsightsPage, RobustnessPage } from "./pages/ProductPages";
import NotFound from "./pages/NotFound";

function Router() {
  return <Switch><Route path="/" component={Home} /><Route path="/analyze" component={Analyze} /><Route path="/history" component={HistoryPage} /><Route path="/insights" component={InsightsPage} /><Route path="/robustness" component={RobustnessPage} /><Route path="/how-it-works" component={HowItWorks} /><Route path="/about" component={About} /><Route path="/404" component={NotFound} /><Route component={NotFound} /></Switch>;
}

function App() {
  return <ErrorBoundary><ThemeProvider defaultTheme="light" switchable><TooltipProvider><Toaster /><SiteHeader /><main><Router /></main><Footer /></TooltipProvider></ThemeProvider></ErrorBoundary>;
}

export default App;

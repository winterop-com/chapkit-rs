import './app.css'

import { StatusProvider } from '@winterop-com/sokkel-ui/StatusProvider'
import { RUN_STATUS_MAP } from '@winterop-com/sokkel-ui/lib/status'
import { Toaster } from '@winterop-com/sokkel-ui/primitives/toast'
import { TooltipProvider } from '@winterop-com/sokkel-ui/primitives/tooltip'
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'

import { App } from './App'

/** A job's states as the service names them, in the run colors. */
const STATUSES = {
    ...RUN_STATUS_MAP,
    pending: { color: 'queued' as const },
    completed: { color: 'succeeded' as const },
    canceling: { color: 'warning' as const },
    canceled: { color: 'cancelled' as const },
    healthy: { color: 'good' as const },
    degraded: { color: 'warning' as const },
    unhealthy: { color: 'critical' as const },
    unreachable: { color: 'critical' as const },
    green: { color: 'good' as const },
    yellow: { color: 'warning' as const },
    orange: { color: 'warning' as const },
    red: { color: 'critical' as const },
}

const root = document.getElementById('root')
if (root === null) throw new Error('No #root element in index.html')

createRoot(root).render(
        <StrictMode>
            <TooltipProvider>
                <StatusProvider statuses={STATUSES}>
                    <Toaster>
                        <App />
                    </Toaster>
                </StatusProvider>
            </TooltipProvider>
        </StrictMode>,
)

//! Webhook alerter (feature `alerting-webhook`).

use super::{Alerter, DriftAlertEvent};
use std::time::Duration;

/// An [`Alerter`] that POSTs a JSON-serialized [`DriftAlertEvent`] to a
/// configured URL.
///
/// This is the generic escape hatch: point it at a Slack incoming webhook, a
/// PagerDuty Events API endpoint, or your own service, and wire drift into any
/// of them without this crate needing a bespoke integration per destination.
///
/// The POST is synchronous (via `reqwest`'s blocking client) to keep the core
/// crate free of an async-runtime requirement. Delivery is best-effort: a failed
/// request is swallowed rather than propagated, since [`Alerter::alert`] cannot
/// fail the drift check itself. Use [`WebhookAlerter::try_send`] if you want the
/// error.
#[derive(Clone, Debug)]
pub struct WebhookAlerter {
    url: String,
    client: reqwest::blocking::Client,
}

impl WebhookAlerter {
    /// Create a webhook alerter targeting `url`, with a default 5-second timeout.
    ///
    /// # Errors
    /// Returns a `reqwest::Error` if the HTTP client cannot be constructed.
    pub fn new(url: impl Into<String>) -> reqwest::Result<Self> {
        Self::with_timeout(url, Duration::from_secs(5))
    }

    /// Create a webhook alerter with an explicit request timeout.
    ///
    /// # Errors
    /// Returns a `reqwest::Error` if the HTTP client cannot be constructed.
    pub fn with_timeout(url: impl Into<String>, timeout: Duration) -> reqwest::Result<Self> {
        let client = reqwest::blocking::Client::builder()
            .timeout(timeout)
            .build()?;
        Ok(Self {
            url: url.into(),
            client,
        })
    }

    /// POST the event and return any transport/HTTP error, instead of swallowing
    /// it as [`Alerter::alert`] does. Useful in tests and when the caller wants
    /// delivery confirmation.
    ///
    /// # Errors
    /// Returns a `reqwest::Error` on a transport failure or non-success status.
    pub fn try_send(&self, event: &DriftAlertEvent) -> reqwest::Result<()> {
        self.client
            .post(&self.url)
            .json(event)
            .send()?
            .error_for_status()?;
        Ok(())
    }
}

impl Alerter for WebhookAlerter {
    fn alert(&self, event: &DriftAlertEvent) {
        // Best-effort: a delivery failure must not fail the drift check.
        let _ = self.try_send(event);
    }
}

"""Phase 19: the client-facing product layer -- turns the single-tenant,
one-hardcoded-email signal pipeline (scaata.live, scaata.notify) into
something a second, third, Nth paying subscriber can actually use.

scaata.product.db          subscriber/watchlist/delivery-idempotency store (SQLite)
scaata.product.billing     Stripe checkout + webhook handling (mock-fallback, same
                            convention as scaata.live.alpaca_broker / scaata.notify.email_sender)
scaata.product.api         FastAPI app: signup, API-key-gated signal reads, Stripe webhook

scaata.live.distribute_signals is the daily fan-out job that ties these
together with the existing scaata.live.daily_signal / scaata.notify.email_sender.
"""

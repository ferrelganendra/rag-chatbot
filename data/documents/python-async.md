# Python Async Programming Guide

## Coroutines

Coroutines are declared with `async def` and are the foundation of async programming in Python. A coroutine is a function that can pause its execution and yield control back to the event loop, allowing other coroutines to run concurrently.

```python
async def fetch_data(url: str) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()
```

## The Event Loop

The event loop is the core of asynchronous execution. It manages and distributes the execution of different tasks. Python's `asyncio` provides a default event loop via `asyncio.run()`.

Key concepts:
- **Tasks**: wrappers around coroutines that schedule them on the event loop
- **Futures**: low-level awaitable objects representing a result that may not be ready yet
- **Callbacks**: functions scheduled to run when a future completes

## Awaitable Objects

Three types of objects can be awaited:
1. **Coroutines** - defined with `async def`
2. **Tasks** - created with `asyncio.create_task()`
3. **Futures** - low-level awaitables

## Common Patterns

### Concurrent Execution with gather
```python
results = await asyncio.gather(
    fetch_data(url1),
    fetch_data(url2),
    fetch_data(url3),
)
```

### Timeouts
```python
try:
    result = await asyncio.wait_for(slow_operation(), timeout=5.0)
except asyncio.TimeoutError:
    print("Operation timed out")
```

### Producer-Consumer with Queues
`asyncio.Queue` is thread-safe and designed for async producer-consumer patterns. Multiple producers can put items while multiple consumers retrieve them, all without locking.

## Error Handling

Unhandled exceptions in tasks don't propagate immediately. Use `task.exception()` to retrieve exceptions from completed tasks. Always wrap concurrent operations in try/except to prevent silent failures.

## Performance Considerations

Async is NOT parallel. It's concurrent. CPU-bound work blocks the event loop. Use `loop.run_in_executor()` to offload CPU-heavy work to a thread pool:

```python
result = await loop.run_in_executor(None, cpu_intensive_function, arg)
```

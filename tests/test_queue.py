import asyncio

from chronos.events.queue import EventQueue
from chronos.protocol import Event, EventType


def ev(t: EventType, n: int = 0) -> Event:
    return Event(session_id="s", type=t, payload={"n": n})


async def test_priority_order():
    q = EventQueue()
    for t in (EventType.TEXT, EventType.CAMERA_FRAME, EventType.TRANSCRIPT_PARTIAL,
              EventType.INTERRUPT):
        await q.put(ev(t))
    got = [(await q.get()).type for _ in range(4)]
    assert got == [EventType.INTERRUPT, EventType.TRANSCRIPT_PARTIAL,
                   EventType.CAMERA_FRAME, EventType.TEXT]


async def test_fifo_within_priority_and_transcript_shared_class():
    q = EventQueue()
    kinds = [EventType.TRANSCRIPT_PARTIAL, EventType.TRANSCRIPT_FINAL] * 5
    for i, t in enumerate(kinds):
        await q.put(ev(t, i))
    assert [(await q.get()).payload["n"] for _ in kinds] == list(range(10))


async def test_late_interrupt_jumps_queue():
    q = EventQueue()
    for i in range(5):
        await q.put(ev(EventType.TEXT, i))
    await q.put(ev(EventType.INTERRUPT, 99))
    assert (await q.get()).payload["n"] == 99


async def test_drain_and_metrics():
    q = EventQueue()
    await q.put(ev(EventType.TEXT, 1))
    await q.put(ev(EventType.INTERRUPT, 2))
    assert q.qsize() == 2 and q.metrics.max_depth == 2
    drained = q.drain()
    assert [e.payload["n"] for e in drained] == [2, 1]
    assert q.empty() and q.drain() == []
    assert q.metrics.puts == 2 and q.metrics.gets == 2
    assert q.metrics.put_by_type[EventType.TEXT] == 1


async def test_get_blocks_until_put():
    q = EventQueue()
    task = asyncio.create_task(q.get())
    await asyncio.sleep(0.01)
    assert not task.done()
    await q.put(ev(EventType.TEXT))
    assert (await asyncio.wait_for(task, 1)).type == EventType.TEXT


async def test_100_concurrent_producers_no_loss_and_per_producer_order():
    q = EventQueue()
    per = 20

    async def producer(pid: int) -> None:
        for i in range(per):
            t = EventType.INTERRUPT if i % 5 == 0 else EventType.TEXT
            await q.put(Event(session_id="s", type=t, payload={"p": pid, "i": i}))
            await asyncio.sleep(0)

    await asyncio.gather(*(producer(p) for p in range(100)))
    assert q.metrics.puts == 100 * per
    out = q.drain()
    assert len(out) == 100 * per
    # all interrupts strictly before any text
    types = [e.type for e in out]
    first_text = types.index(EventType.TEXT)
    assert all(t == EventType.INTERRUPT for t in types[:first_text])
    assert EventType.INTERRUPT not in types[first_text:]
    # FIFO per producer within a priority class
    last: dict[tuple[int, EventType], int] = {}
    for e in out:
        k = (e.payload["p"], e.type)
        assert e.payload["i"] > last.get(k, -1)
        last[k] = e.payload["i"]

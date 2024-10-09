BYTES_PER_INT = 8

def write(f, data):
  assert all([isinstance(d, bytes) for d in data])
  if isinstance(f, str):
    f = open(f, 'wb')
  index_table = []
  for d in data:
    f.write(d)
    if len(index_table) == 0:
      index_table.append(len(d))
    else:
      index_table.append(index_table[-1] + len(d))
  for i in index_table:
    f.write(i.to_bytes(BYTES_PER_INT, 'big'))
  f.write(len(index_table).to_bytes(BYTES_PER_INT, 'big'))


def append(f, data):
  if isinstance(data, bytes):
    data = [data]
  if isinstance(f, str):
    f = open(f, 'rb+')
  index_table = read_index_table(f)
  assert len(index_table) > 0
  # Overwrite index table
  n = len(index_table)
  f.seek(-(n + 1) * BYTES_PER_INT, 2)
  for d in data:
    f.write(d)
    index_table.append(index_table[-1] + len(d))

  # Write index table at the end
  for i in index_table:
    f.write(i.to_bytes(BYTES_PER_INT, 'big'))
  f.write(len(index_table).to_bytes(BYTES_PER_INT, 'big'))


def read_len(f):
  if isinstance(f, str):
    f = open(f, 'rb')
  f.seek(-BYTES_PER_INT, 2)
  n = int.from_bytes(f.read(BYTES_PER_INT), 'big')
  return n


def read_index_table(f):
  if isinstance(f, str):
    f = open(f, 'rb')
  f.seek(-BYTES_PER_INT, 2)
  n = int.from_bytes(f.read(BYTES_PER_INT), 'big')
  f.seek(-(n + 1) * BYTES_PER_INT, 2)
  index_table = []
  for _ in range(n):
    i = int.from_bytes(f.read(BYTES_PER_INT), 'big')
    index_table.append(i)
  return index_table


def read_data(f, idx):
  if isinstance(f, str):
    f = open(f, 'rb')
  f.seek(-BYTES_PER_INT, 2)
  n = int.from_bytes(f.read(BYTES_PER_INT), 'big')
  assert 0 <= idx < n,  \
      f"Index {idx} out of bounds for file with {n} elements"
  f.seek(-(n - idx + 1) * BYTES_PER_INT, 2)
  end = int.from_bytes(f.read(BYTES_PER_INT), 'big')
  if idx - 1 < 0:
    start = 0
  else:
    f.seek(-2 * BYTES_PER_INT, 1)
    start = int.from_bytes(f.read(BYTES_PER_INT), 'big')
  f.seek(start, 0)
  return f.read(end - start)


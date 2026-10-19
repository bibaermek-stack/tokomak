-- Numbered display equations for the Word build: a borderless one-row table [ blank | equation | (n) ].
-- The number stays at the right margin; docx_post.py removes the table borders.
function Div(el)
  if not el.classes:includes('eqn') then return nil end
  local num = el.attributes['num']
  local body = {}
  for _, b in ipairs(el.content) do
    if b.t == 'Para' then body[#body + 1] = pandoc.Plain(b.content) else body[#body + 1] = b end
  end
  local st = pandoc.SimpleTable(
    {},                                                         -- caption
    { 'AlignLeft', 'AlignCenter', 'AlignRight' },
    { 0.07, 0.86, 0.07 },
    {},                                                         -- no header
    { { {}, body, { pandoc.Plain({ pandoc.Str('(' .. num .. ')') }) } } }
  )
  return pandoc.utils.from_simple_table(st)
end

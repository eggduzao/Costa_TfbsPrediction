import smith
from smith import Tensor


def milstm_cell(x, hx, cx, w_ih, w_hh, alpha, beta_i, beta_h, bias):
    Wx = x.mm(w_ih.t())
    Uz = hx.mm(w_hh.t())

    # Section 2.1 in https://arxiv.org/pdf/1606.06630.pdf
    gates = alpha * Wx * Uz + beta_i * Wx + beta_h * Uz + bias

    # Same as LSTMCell after this point
    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = ingate.sigmoid()
    forgetgate = forgetgate.sigmoid()
    cellgate = cellgate.tanh()
    outgate = outgate.sigmoid()

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * cy.tanh()

    return hy, cy


def lstm_cell(
    input: Tensor,
    hidden: tuple[Tensor, Tensor],
    w_ih: Tensor,
    w_hh: Tensor,
    b_ih: Tensor,
    b_hh: Tensor,
) -> tuple[Tensor, Tensor]:
    hx, cx = hidden
    gates = smith.mm(input, w_ih.t()) + smith.mm(hx, w_hh.t()) + b_ih + b_hh

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = smith.sigmoid(ingate)
    forgetgate = smith.sigmoid(forgetgate)
    cellgate = smith.tanh(cellgate)
    outgate = smith.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * smith.tanh(cy)

    return hy, cy


def flat_lstm_cell(
    input: Tensor,
    hx: Tensor,
    cx: Tensor,
    w_ih: Tensor,
    w_hh: Tensor,
    b_ih: Tensor,
    b_hh: Tensor,
) -> tuple[Tensor, Tensor]:
    gates = smith.mm(input, w_ih.t()) + smith.mm(hx, w_hh.t()) + b_ih + b_hh

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = smith.sigmoid(ingate)
    forgetgate = smith.sigmoid(forgetgate)
    cellgate = smith.tanh(cellgate)
    outgate = smith.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * smith.tanh(cy)

    return hy, cy


def premul_lstm_cell(
    igates: Tensor,
    hidden: tuple[Tensor, Tensor],
    w_hh: Tensor,
    b_ih: Tensor,
    b_hh: Tensor,
) -> tuple[Tensor, Tensor]:
    hx, cx = hidden
    gates = igates + smith.mm(hx, w_hh.t()) + b_ih + b_hh

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = smith.sigmoid(ingate)
    forgetgate = smith.sigmoid(forgetgate)
    cellgate = smith.tanh(cellgate)
    outgate = smith.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * smith.tanh(cy)

    return hy, cy


def premul_lstm_cell_no_bias(
    igates: Tensor, hidden: tuple[Tensor, Tensor], w_hh: Tensor, b_hh: Tensor
) -> tuple[Tensor, Tensor]:
    hx, cx = hidden
    gates = igates + smith.mm(hx, w_hh.t()) + b_hh

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = smith.sigmoid(ingate)
    forgetgate = smith.sigmoid(forgetgate)
    cellgate = smith.tanh(cellgate)
    outgate = smith.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * smith.tanh(cy)

    return hy, cy


def gru_cell(input, hidden, w_ih, w_hh, b_ih, b_hh):
    gi = smith.mm(input, w_ih.t()) + b_ih
    gh = smith.mm(hidden, w_hh.t()) + b_hh
    i_r, i_i, i_n = gi.chunk(3, 1)
    h_r, h_i, h_n = gh.chunk(3, 1)

    resetgate = smith.sigmoid(i_r + h_r)
    inputgate = smith.sigmoid(i_i + h_i)
    newgate = smith.tanh(i_n + resetgate * h_n)
    hy = newgate + inputgate * (hidden - newgate)

    return hy


def rnn_relu_cell(input, hidden, w_ih, w_hh, b_ih, b_hh):
    igates = smith.mm(input, w_ih.t()) + b_ih
    hgates = smith.mm(hidden, w_hh.t()) + b_hh
    return smith.relu(igates + hgates)


def rnn_tanh_cell(input, hidden, w_ih, w_hh, b_ih, b_hh):
    igates = smith.mm(input, w_ih.t()) + b_ih
    hgates = smith.mm(hidden, w_hh.t()) + b_hh
    return smith.tanh(igates + hgates)

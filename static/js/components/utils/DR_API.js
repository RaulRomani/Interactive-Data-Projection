const api = process.env.REACT_APP_CONTACTS_API_URL || 'http://localhost:5000'

let token = localStorage.token

if (!token)
  token = localStorage.token = Math.random().toString(36).substr(-8)

const headers = {
  'Accept': 'application/json',
  'Authorization': token
}

/*export const getControlPoints = () =>
  fetch(`${api}/controlPoints`, { headers })
    .then(res => res.json())
    .then(data => data)*/

export const getControlPoints = (body) =>
  fetch(`${api}/controlPoints`, {
    method: 'POST',
    headers: {
      'Accept': 'application/json',
  	  'Authorization': token,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(body)
  }).then(res => res.json())

export const projectUsingEnsemble = (body) =>
  fetch(`${api}/projectUsingEnsemble`, {
    method: 'POST',
    headers: {
      'Accept': 'application/json',
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(body)
  }).then(res => res.json())

export const estimateEnsemble = (body) =>
  fetch(`${api}/estimateEnsemble`, {
    method: 'POST',
    headers: {
      'Accept': 'application/json',
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(body)
  }).then(res => res.json())

/*
export const getAll = () =>
  fetch(`${api}/contacts`, { headers })
    .then(res => res.json())
    .then(data => data.contacts)

export const remove = (contact) =>
  fetch(`${api}/contacts/${contact.id}`, { method: 'DELETE', headers })
    .then(res => res.json())
    .then(data => data.contact)

export const create = (body) =>
  fetch(`${api}/contacts`, {
    method: 'POST',
    headers: {
      ...headers,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(body)
  }).then(res => res.json())
*/
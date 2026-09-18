"""Delivery channels for a generated ``Signal`` (approved plan point 8).
Every provider implements the same ``NotificationProvider`` interface, so
``signals.service.send_notification`` -- and everything upstream of it --
never needs to change when the channel does. No provider here ever
places a trade or talks to a broker; a provider's only job is to make a
signal visible to a person.
"""
